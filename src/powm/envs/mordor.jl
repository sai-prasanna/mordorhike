import Pkg
# Pkg.add("POMDPs")
# Pkg.add("POMCPOW")
# Pkg.add("POMDPModels")
# Pkg.add("POMDPTools")
# Pkg.add("Distributions")
# Pkg.add("Images")
# Pkg.add("ImageView")
# Pkg.add("TestImages")
# Pkg.add("Colors")
# Pkg.add("GLMakie")
# Pkg.add("ImageDraw")
#Pkg.add("DataStructures")
import POMDPs
import POMDPModels
import Distributions

using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using LinearAlgebra
using Distributions
using Random
using Distributions: VonMises
using DataStructures


using Images
using ImageView
using TestImages
using Colors
using GLMakie
using ImageDraw
using GLMakie.GLFW
using GLMakie: Axis, Figure, to_native, display
using ImageDraw: Point, LineSegment, CirclePointRadius

@enum LateralAction strafe rotate



# Update the struct definition to include new fields
mutable struct MordorHikePOMDP <: POMDP{Vector{Float64}, Int64, Vector{Float64}}
    # Constants
    translate_step::Float64
    translate_std::Float64
    obs_std::Float64
    discount::Float64
    
    # Map bounds and positions
    map_lower_bound::Vector{Float64}
    map_upper_bound::Vector{Float64}
    fixed_start_pos::Vector{Float64}
    goal_position::Vector{Float64}
    
    # Mountain parameters
    slope::Vector{Float64}
    mvn_1::MvNormal
    mvn_2::MvNormal
    mvn_3::MvNormal

    # New fields
    occlude_dims::Tuple{Int, Int}
    start_distribution::String
    lateral_action::LateralAction
    rotate_step::Float64
    rotate_kappa::Union{Float64, Nothing}
    action_failure_prob::Float64
    uniform_start_lower_bound::Vector{Float64}
    uniform_start_upper_bound::Vector{Float64}

    # Constructor
    function MordorHikePOMDP(;
        occlude_dims=(1, 2),
        translate_step=0.1,
        rotate_step=π/2,
        translate_std=0.05,
        rotate_kappa=nothing,
        action_failure_prob=0.0,
        start_distribution="fixed",
        obs_std=0.1,
        lateral_action=strafe)
        
        new(
            translate_step,    # translate_step
            translate_std,     # translate_std
            obs_std,          # obs_std
            0.99,             # discount
            [-1.0, -1.0],     # map_lower_bound
            [1.0, 1.0],       # map_upper_bound
            [-0.8, -0.8],     # fixed_start_pos
            [0.8, 0.8],       # goal_position
            [0.2, 0.2],       # slope
            MvNormal([0.0, 0.0], [0.005 0.0; 0.0 1.0]),    # mvn_1
            MvNormal([0.0, -0.8], [1.0 0.0; 0.0 0.01]),    # mvn_2
            MvNormal([0.0, 0.8], [1.0 0.0; 0.0 0.01]),     # mvn_3
            occlude_dims,         # occlude_dims
            start_distribution,    # start_distribution
            lateral_action,        # lateral_action
            rotate_step,          # rotate_step
            rotate_kappa,         # rotate_kappa
            action_failure_prob,  # action_failure_prob
            [-1.0, -1.0],        # uniform_start_lower_bound
            [1.0, 0.0]           # uniform_start_upper_bound
        )
    end    
end

# State: [x, y, θ]
# Action: 1=north, 2=south, 3=east, 4=west
# Observation: [altitude]

function POMDPs.actions(pomdp::MordorHikePOMDP)
    return [1, 2, 3, 4]
end

# Update initialstate to handle different start distributions
function POMDPs.initialstate(pomdp::MordorHikePOMDP)
    return ImplicitDistribution(function(rng)
        if pomdp.start_distribution == "fixed"
            return [pomdp.fixed_start_pos[1], pomdp.fixed_start_pos[2], 0.0]
        elseif pomdp.start_distribution == "rotation"
            theta = rand(rng, [0.0, π/2, π, 3π/2])
            return [pomdp.fixed_start_pos[1], pomdp.fixed_start_pos[2], theta]
        elseif pomdp.start_distribution == "uniform"
            pos = rand(rng, 2) .* (pomdp.uniform_start_upper_bound - pomdp.uniform_start_lower_bound) + pomdp.uniform_start_lower_bound
            theta = rand(rng, [0.0, π/2, π, 3π/2])
            return [pos[1], pos[2], theta]
        else
            error("Invalid start distribution")
        end
    end)
end

function POMDPs.reward(pomdp::MordorHikePOMDP, s::Vector{Float64}, a::Int64, sp::Vector{Float64})
    if isterminal(pomdp, s)
        return 0.0
    else
        return calculate_altitude(pomdp, sp[1:2])
    end
end
POMDPs.discount(pomdp::MordorHikePOMDP) = pomdp.discount

function POMDPs.isterminal(pomdp::MordorHikePOMDP, s::Vector{Float64})
    return norm(s[1:2] - pomdp.goal_position) <= 2 * pomdp.translate_step
end

function get_horizon(pomdp::MordorHikePOMDP)
    max_manhattan_distance = sum(abs.(pomdp.map_upper_bound - pomdp.map_lower_bound))
    
    factor = if pomdp.start_distribution == "fixed"
        1
    elseif pomdp.start_distribution == "rotation"
        2
    else  # uniform
        4
    end
    
    return factor * Int(ceil(max_manhattan_distance / pomdp.translate_step)) * 2
end

function POMDPs.transition(pomdp::MordorHikePOMDP, s::Vector{Float64}, a::Int64)
    return ImplicitDistribution(function(rng)
        # Check for action failure
        if rand(rng) < pomdp.action_failure_prob
            return s
        end
        
        theta = s[3]
        
        # Create movement vector based on theta
        forward_vector = [cos(theta), sin(theta)]
        lateral_vector = [-sin(theta), cos(theta)]
        
        # Map actions to movement/rotation
        if pomdp.lateral_action == strafe
            movement = if a == 1  # North
                forward_vector
            elseif a == 2  # South 
                -forward_vector
            elseif a == 3  # East
                lateral_vector
            else  # West
                -lateral_vector
            end
            
            next_pos = s[1:2] + movement * pomdp.translate_step
            next_pos += rand(rng, Normal(0, pomdp.translate_std), 2)
            next_pos = clamp.(next_pos, pomdp.map_lower_bound, pomdp.map_upper_bound)
            next_theta = theta
            
        else  # rotate mode
            if a == 1  # Forward
                next_pos = s[1:2] + forward_vector * pomdp.translate_step
                next_theta = theta
            elseif a == 2  # Backward
                next_pos = s[1:2] - forward_vector * pomdp.translate_step
                next_theta = theta
            elseif a == 3  # Turn right
                next_pos = s[1:2]
                next_theta = theta + pomdp.rotate_step
            else  # Turn left
                next_pos = s[1:2]
                next_theta = theta - pomdp.rotate_step
            end
            
            next_pos += rand(rng, Normal(0, pomdp.translate_std), 2)
            next_pos = clamp.(next_pos, pomdp.map_lower_bound, pomdp.map_upper_bound)
            
            # Apply von Mises noise to rotation if specified
            if !isnothing(pomdp.rotate_kappa)
                next_theta += rand(rng, VonMises(0, pomdp.rotate_kappa))
            end
        end
        
        next_theta = mod(next_theta, 2π)
        return [next_pos[1], next_pos[2], next_theta]
    end)
end


function POMDPs.observation(pomdp::MordorHikePOMDP, a::Int64, sp::Vector{Float64})
    # Create full observation
    position = sp[1:2]
    altitude = calculate_altitude(pomdp, position)
    full_obs = vcat(position, altitude)
    
    # Apply noise
    full_obs += rand(Normal(0, pomdp.obs_std), length(full_obs))
    
    # Occlude dimensions
    obs = deleteat!(copy(full_obs), collect(pomdp.occlude_dims))
    
    return MvNormal(obs, Matrix(I, length(obs), length(obs)) * pomdp.obs_std)
end

function POMDPs.observation(pomdp::MordorHikePOMDP, s::Vector{Float64}, a::Int64, sp::Vector{Float64})
    return observation(pomdp, a, sp)
end


function calculate_altitude(pomdp::MordorHikePOMDP, pos::Vector{Float64})
    # Convert pos to a 2-element vector to ensure correct dimensionality
    pos_2d = pos[1:2]  # Take only x,y coordinates if pos is longer
    mountains = [
        pdf(pomdp.mvn_1, pos_2d),
        pdf(pomdp.mvn_2, pos_2d),
        pdf(pomdp.mvn_3, pos_2d)
    ]
    altitude = maximum(mountains)
    return -exp(-altitude) + dot(pos_2d, pomdp.slope) - 0.02
end


struct GoToGoalPolicy{P} <: Policy
    pomdp::P
end

function POMDPs.action(p::GoToGoalPolicy, s)
    # Get vector to goal
    goal_vec = p.pomdp.goal_position - s[1:2]

    # Get current orientation vectors
    theta = s[3]
    forward = [cos(theta), sin(theta)]
    lateral = [-sin(theta), cos(theta)]
    
    # Calculate dot products with goal vector
    forward_alignment = dot(normalize(goal_vec), forward)
    lateral_alignment = dot(normalize(goal_vec), lateral)
    
    # Choose action that best aligns with goal
    # Actions are: 1=North (forward), 2=South (backward), 
    #             3=East (right), 4=West (left)
    if abs(forward_alignment) > abs(lateral_alignment)
        if forward_alignment > 0
            return 1 # Move forward
        else
            return 2 # Move backward
        end
    else
        if lateral_alignment > 0
            return 3 # Move right
        else
            return 4 # Move left
        end
    end
end

function MordorHikePOMDP(::Val{:easy}; kwargs...)
    return MordorHikePOMDP(;
        occlude_dims=(1, 2),
        start_distribution="fixed",
        kwargs...
    )
end

function MordorHikePOMDP(::Val{:medium}; kwargs...)
    return MordorHikePOMDP(;
        occlude_dims=(1, 2),
        start_distribution="rotation",
        kwargs...
    )
end

function MordorHikePOMDP(::Val{:hard}; kwargs...)
    return MordorHikePOMDP(;
        occlude_dims=(1, 2),
        start_distribution="uniform",
        kwargs...
    )
end

function MordorHikePOMDP(::Val{:veryhard}; kwargs...)
    return MordorHikePOMDP(;
        occlude_dims=(1, 2),
        start_distribution="rotation",
        lateral_action=rotate,
        kwargs...
    )
end

function world_to_pixel(pomdp::MordorHikePOMDP, coord::Vector{Float64}, render_size::Tuple{Int,Int})
    x = round(Int, (coord[1] - pomdp.map_lower_bound[1]) / 
        (pomdp.map_upper_bound[1] - pomdp.map_lower_bound[1]) * render_size[2])
    
    # Flip y-axis for image coordinates
    y = round(Int, (1 - (coord[2] - pomdp.map_lower_bound[2]) / 
        (pomdp.map_upper_bound[2] - pomdp.map_lower_bound[2])) * render_size[1])
    
    return [x, y]
end

function create_background(pomdp::MordorHikePOMDP, render_size::Tuple{Int,Int})
    height, width = render_size
    x = range(pomdp.map_lower_bound[1], pomdp.map_upper_bound[1], length=width)
    y = range(pomdp.map_lower_bound[1], pomdp.map_upper_bound[1], length=height)
    
    Z = zeros(height, width)
    for i in 1:height, j in 1:width
        pos = [x[j], y[i]]
        Z[i,j] = calculate_altitude(pomdp, pos)
    end
    
    # Normalize Z to 0-1 range
    Z_norm = (Z .- minimum(Z)) ./ (maximum(Z) - minimum(Z))
    
    # Create color map using viridis-like colors
    img = RGB.(Z_norm, Z_norm, 1 .- Z_norm)
    
    return img
end

function render(pomdp::MordorHikePOMDP, s::Vector{Float64}; 
    render_size::Tuple{Int,Int}=(128,128), path::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    # Create background if not cached
    if !@isdefined(background)
        background = create_background(pomdp, render_size)
    end
    
    # Create a copy of the background to draw on
    img = copy(background)
    
    # Draw path
    if length(path) > 1
        for i in 2:length(path)
            p1 = world_to_pixel(pomdp, path[i-1], render_size)
            p2 = world_to_pixel(pomdp, path[i], render_size)
            # Draw line between p1 and p2 in red
            draw!(img, LineSegment(Point(round(Int, p1[1]), round(Int, p1[2])), 
                                 Point(round(Int, p2[1]), round(Int, p2[2]))), 
                  RGB{Float64}(1,0,0))
        end
    end
    
    # Draw goal
    goal_pixel = world_to_pixel(pomdp, pomdp.goal_position, render_size)
    draw!(img, CirclePointRadius(Point(round(Int, goal_pixel[1]), round(Int, goal_pixel[2])), 4), 
          RGB{Float64}(0,0,1))
    
    # Draw current position and direction
    pos_pixel = world_to_pixel(pomdp, s[1:2], render_size)
    theta = s[3]
    direction = [
        pos_pixel[1] + 7*cos(theta),
        pos_pixel[2] - 7*sin(theta)
    ]
    
    # Draw position circle
    draw!(img, CirclePointRadius(Point(round(Int, pos_pixel[1]), round(Int, pos_pixel[2])), 5), 
          RGB{Float64}(1,0,0))
    # Draw direction line
    draw!(img, LineSegment(Point(round(Int, pos_pixel[1]), round(Int, pos_pixel[2])), 
                          Point(round(Int, direction[1]), round(Int, direction[2]))), 
          RGB{Float64}(1,0,0))
    
    return img
end

# Add interactive visualization function
function play_interactive()
    pomdp = MordorHikePOMDP(Val(:medium))
    s = rand(initialstate(pomdp))
    path = [s[1:2]]
    total_reward = 0.0
    
    # Create figure and image plot
    fig = Figure(size=(300, 300))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Initial render
    img = render(pomdp, s, path=path)
    image!(ax, rotr90(img))
    
    # Get the GLFW window
    glfw_window = to_native(display(fig))
    
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
            action = if event.key == Keyboard.w
                1
            elseif event.key == Keyboard.s
                2
            elseif event.key == Keyboard.a
                3
            elseif event.key == Keyboard.d
                4
            elseif event.key == Keyboard.q
                println("Total return: $total_reward")
                GLFW.SetWindowShouldClose(glfw_window, true)
                return false
            else
                return true
            end
            
            # Update state
            sp = rand(transition(pomdp, s, action))
            o = rand(observation(pomdp, s, action, sp))
            r = reward(pomdp, s, action, sp)
            total_reward += r
            
            println("Observation: $o")
            println("Reward: $r")
            
            push!(path, sp[1:2])
            s = sp
            
            # Update visualization
            img = render(pomdp, s, path=path)
            image!(ax, rotr90(img))
            
            # Check if terminal
            if isterminal(pomdp, s)
                println("Goal reached!")
                println("Total return: $total_reward")
                GLFW.SetWindowShouldClose(glfw_window, true)
            end
            return true
        end
        return true
    end
    
    # Wait for window to close
    while !GLFW.WindowShouldClose(glfw_window)
        sleep(0.01)
    end
end



struct AStarRolloutPolicy{P} <: Policy
    pomdp::P
    grid_size::Int
    cost_cache::Dict{Tuple{Int,Int}, Float64}
    
    function AStarRolloutPolicy(pomdp::P, grid_size::Int=20) where P
        cost_cache = Dict{Tuple{Int,Int}, Float64}()
        new{P}(pomdp, grid_size, cost_cache)
    end
end

function world_to_grid(policy::AStarRolloutPolicy, pos::Vector{Float64})
    pomdp = policy.pomdp
    x_norm = (pos[1] - pomdp.map_lower_bound[1]) / (pomdp.map_upper_bound[1] - pomdp.map_lower_bound[1])
    y_norm = (pos[2] - pomdp.map_lower_bound[2]) / (pomdp.map_upper_bound[2] - pomdp.map_lower_bound[2])
    
    grid_x = round(Int, x_norm * (policy.grid_size - 1)) + 1
    grid_y = round(Int, y_norm * (policy.grid_size - 1)) + 1
    
    return (clamp(grid_x, 1, policy.grid_size), clamp(grid_y, 1, policy.grid_size))
end

function grid_to_world(policy::AStarRolloutPolicy, grid_pos::Tuple{Int,Int})
    pomdp = policy.pomdp
    x_norm = (grid_pos[1] - 1) / (policy.grid_size - 1)
    y_norm = (grid_pos[2] - 1) / (policy.grid_size - 1)
    
    x = x_norm * (pomdp.map_upper_bound[1] - pomdp.map_lower_bound[1]) + pomdp.map_lower_bound[1]
    y = y_norm * (pomdp.map_upper_bound[2] - pomdp.map_lower_bound[2]) + pomdp.map_lower_bound[2]
    
    return [x, y]
end

function get_neighbors(pos::Tuple{Int,Int}, grid_size::Int)
    neighbors = Tuple{Int,Int}[]
    for (dx, dy) in [(0,1), (0,-1), (1,0), (-1,0)]
        new_x = pos[1] + dx
        new_y = pos[2] + dy
        if 1 ≤ new_x ≤ grid_size && 1 ≤ new_y ≤ grid_size
            push!(neighbors, (new_x, new_y))
        end
    end
    return neighbors
end

function heuristic(pos::Tuple{Int,Int}, goal::Tuple{Int,Int})
    return abs(pos[1] - goal[1]) + abs(pos[2] - goal[2])
end

function get_path_to_goal(policy::AStarRolloutPolicy, start_pos::Vector{Float64})
    pomdp = policy.pomdp
    start_grid = world_to_grid(policy, start_pos)
    goal_grid = world_to_grid(policy, pomdp.goal_position)
    
    # A* implementation
    frontier = PriorityQueue{Tuple{Int,Int}, Float64}()
    enqueue!(frontier, start_grid => 0.0)
    
    came_from = Dict{Tuple{Int,Int}, Union{Nothing, Tuple{Int,Int}}}()
    cost_so_far = Dict{Tuple{Int,Int}, Float64}()
    
    came_from[start_grid] = nothing
    cost_so_far[start_grid] = 0.0
    
    while !isempty(frontier)
        current = dequeue!(frontier)
        
        if current == goal_grid
            break
        end
        
        for next in get_neighbors(current, policy.grid_size)
            world_pos = grid_to_world(policy, next)
            new_cost = cost_so_far[current] - calculate_altitude(pomdp, world_pos)
            
            if !haskey(cost_so_far, next) || new_cost < cost_so_far[next]
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal_grid)
                frontier[next] = priority
                came_from[next] = current
            end
        end
    end
    
    # Reconstruct path
    path = Vector{Float64}[]
    current = goal_grid
    while current != nothing
        pushfirst!(path, grid_to_world(policy, current))
        current = get(came_from, current, nothing)
    end
    
    return path
end

function POMDPs.action(policy::AStarRolloutPolicy, s)
    path = get_path_to_goal(policy, s[1:2])
    
    if length(path) < 2
        return rand(1:4)  # Random action if no path found
    end
    
    # Get vector to next waypoint
    next_pos = path[min(2, length(path))]
    goal_vec = next_pos - s[1:2]
    
    # Get current orientation vectors
    theta = s[3]
    forward = [cos(theta), sin(theta)]
    lateral = [-sin(theta), cos(theta)]
    
    # Calculate dot products with goal vector
    forward_alignment = dot(normalize(goal_vec), forward)
    lateral_alignment = dot(normalize(goal_vec), lateral)
    
    # Choose action based on alignment
    if abs(forward_alignment) > abs(lateral_alignment)
        return forward_alignment > 0 ? 1 : 2  # Forward or backward
    else
        return lateral_alignment > 0 ? 3 : 4  # Right or left
    end
end


#play_interactive()


pomdp = MordorHikePOMDP(Val(:medium))
# Define solvers

policy = Dict(
    #"POMCPOW (deeper)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=100000, max_depth=100), pomdp),
    "A* Rollout" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(AStarRolloutPolicy(pomdp, 40))), pomdp),
    #"POMCPOW" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10000, max_depth=10), pomdp),
    #"POMCPOW - GoToGoalPolicy" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10000, max_depth=10, estimate_value=FORollout(GoToGoalPolicy(pomdp))), pomdp),
    #"Random" => RandomPolicy(pomdp)
)

# Simulate
hr = HistoryRecorder(max_steps=get_horizon(pomdp))

# Run experiments for each solver
for (policy_name, planner) in policy
    
    println("\n$policy_name Policy")
    returns = []
    scores = []
    lengths = []
    for i in 1:10
        hist = simulate(hr, pomdp, planner) 
        push!(returns, discounted_reward(hist))
        push!(scores, sum(r for (s, b, a, r, sp, o) in hist))
        push!(lengths, length(hist))
        println("Return: $(returns[i]), Score: $(scores[i]), Length: $(lengths[i])")
        # display the path
        path = [s[1:2] for (s, b, a, r, sp, o) in hist]
        last_state = hist[end][1]
        display(render(pomdp, last_state, path=path))
        sleep(2.0)
    end

    println(""" $policy_name Policy
        Mean and std of returns: $(mean(returns)), $(std(returns))
        Mean and std of scores: $(mean(scores)), $(std(scores))
        Mean and std of lengths: $(mean(lengths)), $(std(lengths))
        """)
end

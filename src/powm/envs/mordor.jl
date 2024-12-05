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
# Pkg.add("DataStructures")
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
    render_size::Tuple{Int,Int}=(128,128), 
    path::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
    beliefs::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
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
            draw!(img, LineSegment(Point(round(Int, p1[1]), round(Int, p1[2])), 
                                 Point(round(Int, p2[1]), round(Int, p2[2]))), 
                  RGB{Float64}(1,0,0))
        end
    end
    
    # Draw beliefs if provided
    if !isempty(beliefs)
        for belief in beliefs
            pos_pixel = world_to_pixel(pomdp, belief[1:2], render_size)
            theta = belief[3]
            direction = [
                pos_pixel[1] + 5*cos(theta),  # Shorter arrow for beliefs
                pos_pixel[2] - 5*sin(theta)
            ]
            
            # Draw smaller position circle for belief
            draw!(img, CirclePointRadius(Point(round(Int, pos_pixel[1]), round(Int, pos_pixel[2])), 2), 
                 RGB{Float64}(0.8,0.2,0.8))  # Purple color for beliefs
            # Draw direction line
            draw!(img, LineSegment(Point(round(Int, pos_pixel[1]), round(Int, pos_pixel[2])), 
                                 Point(round(Int, direction[1]), round(Int, direction[2]))), 
                 RGB{Float64}(0.8,0.2,0.8))
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
    pomdp = MordorHikePOMDP(Val(:veryhard))
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

# Create grid value policy
struct GridValuePolicy{P} <: Policy
    pomdp::P
    grid_size::Int
    theta_grid_size::Int
    value_grid::Array{Float64, 3}
    policy_grid::Array{Int, 3}
    
    function GridValuePolicy(pomdp::P, grid_size::Int=100, theta_grid_size::Int=4, max_iter::Int=2000) where P
        value_grid, policy_grid = compute_value_iteration(pomdp, grid_size, theta_grid_size, max_iter=max_iter)
        new{P}(pomdp, grid_size, theta_grid_size, value_grid, policy_grid)
    end
end

function state_to_discrete(s::Vector{Float64}, upper_bound::Vector{Float64}, lower_bound::Vector{Float64}, grid_size::Int, theta_grid_size::Int)
    # Calculate cell size
    x_cell_size = (upper_bound[1] - lower_bound[1]) / grid_size
    y_cell_size = (upper_bound[2] - lower_bound[2]) / grid_size
    theta_cell_size = 2π / theta_grid_size
    
    # Calculate grid indices (floor + 1 ensures 1-based indexing)
    i = clamp(floor(Int, (s[1] - lower_bound[1]) / x_cell_size) + 1, 1, grid_size)
    j = clamp(floor(Int, (s[2] - lower_bound[2]) / y_cell_size) + 1, 1, grid_size)
    k = clamp(floor(Int, mod(s[3], 2π) / theta_cell_size) + 1, 1, theta_grid_size)
    
    return (i, j, k)
end

function discrete_to_state(discrete_state::Tuple{Int,Int,Int}, upper_bound::Vector{Float64}, lower_bound::Vector{Float64}, grid_size::Int, theta_grid_size::Int)
    # Calculate cell size
    x_cell_size = (upper_bound[1] - lower_bound[1]) / grid_size
    y_cell_size = (upper_bound[2] - lower_bound[2]) / grid_size
    theta_cell_size = 2π / theta_grid_size
    
    # Return center of the cell
    x = lower_bound[1] + (discrete_state[1] - 0.5) * x_cell_size
    y = lower_bound[2] + (discrete_state[2] - 0.5) * y_cell_size
    theta = (discrete_state[3] - 0.5) * theta_cell_size
    
    return [x, y, theta]
end


function compute_value_iteration(pomdp::MordorHikePOMDP, grid_size::Int, theta_grid_size::Int;
                                epsilon::Float64=1e-6, max_iter::Int=10000)
    # Initialize value and policy grids
    value_grid = fill(0.0, grid_size, grid_size, theta_grid_size)
    policy_grid = zeros(Int, grid_size, grid_size, theta_grid_size)
    
    # Create world coordinate mappings
    # x_coords = range(pomdp.map_lower_bound[1], pomdp.map_upper_bound[1], length=grid_size)
    # y_coords = range(pomdp.map_lower_bound[2], pomdp.map_upper_bound[2], length=grid_size)
    # theta_coords = range(0, 2π, length=theta_grid_size)
    
    # Value iteration
    for iter in 1:max_iter
        delta = 0.0
        
        for i in 1:grid_size, j in 1:grid_size, k in 1:theta_grid_size
            current_state = discrete_to_state((i, j, k), pomdp.map_upper_bound, pomdp.map_lower_bound, grid_size, theta_grid_size)
            
            # Skip if terminal
            if isterminal(pomdp, current_state)
                value_grid[i, j, k] = 0.0
                continue
            end
            
            current_value = value_grid[i, j, k]
            max_value = -Inf
            best_action = 1
            
            # Try each action
            for action in actions(pomdp)
                sp_dist = transition(pomdp, current_state, action)
                new_value = 0.0
                
                # Sample next states
                n_samples = 10
                for _ in 1:n_samples
                    sp = rand(sp_dist)
                    
                    # Get grid indices for next state
                    next_discrete_state = state_to_discrete(sp, pomdp.map_upper_bound, pomdp.map_lower_bound, grid_size, theta_grid_size)
                    
                    # Compute reward and next value
                    r = reward(pomdp, current_state, action, sp)
                    next_value = isterminal(pomdp, sp) ? 0.0 : value_grid[next_discrete_state...]
                    
                    new_value += (r + pomdp.discount * next_value) / n_samples
                end
                
                if new_value > max_value
                    max_value = new_value
                    best_action = action
                end
            end
            value_grid[i, j, k] = max_value
            policy_grid[i, j, k] = best_action
            delta = max(delta, abs(max_value - current_value))
        end
        
        if delta < epsilon
            println("Value iteration converged after $iter iterations")
            break
        end
        
        if iter % 100 == 0
            println("Iteration $iter, delta = $delta")
        end
    end
    
    return value_grid, policy_grid
end

function POMDPs.action(policy::GridValuePolicy, s)
    # # Convert state position to grid indices
    # x_norm = (s[1] - policy.pomdp.map_lower_bound[1]) / 
    #          (policy.pomdp.map_upper_bound[1] - policy.pomdp.map_lower_bound[1])
    # y_norm = (s[2] - policy.pomdp.map_lower_bound[2]) / 
    #          (policy.pomdp.map_upper_bound[2] - policy.pomdp.map_lower_bound[2])
    # theta_norm = (s[3] - 0) / (2π)
    
    # i = round(Int, x_norm * (policy.grid_size - 1)) + 1
    # j = round(Int, y_norm * (policy.grid_size - 1)) + 1
    # k = round(Int, theta_norm * (policy.theta_grid_size - 1)) + 1
    # # Clamp indices to valid range
    # i = clamp(i, 1, policy.grid_size)
    # j = clamp(j, 1, policy.grid_size)
    # k = clamp(k, 1, policy.theta_grid_size)
    # Get the grid policy action
    return policy.policy_grid[state_to_discrete(s, pomdp.map_upper_bound, pomdp.map_lower_bound, policy.grid_size, policy.theta_grid_size)...]
end

struct GoToGoalPolicy <: Policy
    pomdp::MordorHikePOMDP
end

function POMDPs.action(policy::GoToGoalPolicy, s)
    # Get vector from current position to goal
    goal_vector = policy.pomdp.goal_position - s[1:2]
    
    # Get current heading vector based on theta
    heading = [cos(s[3]), sin(s[3])]
    
    # Get vector perpendicular to heading (for lateral movement)
    perp_heading = [-sin(s[3]), cos(s[3])]
    
    # Project goal vector onto heading and perpendicular directions
    forward_component = dot(goal_vector, heading)
    lateral_component = dot(goal_vector, perp_heading)
    
    # Choose action based on largest component
    if abs(forward_component) > abs(lateral_component)
        # Move forward/backward
        return forward_component > 0 ? 1 : 2  # 1=forward, 2=backward
    else
        # Move laterally
        return lateral_component > 0 ? 3 : 4  # 3=right, 4=left
    end
end


struct QValuePolicy{P} <: Policy
    pomdp::P
    grid_size::Int
    theta_grid_size::Int
    q_table::Array{Float64, 4}  # (x, y, theta, action)
    
    function QValuePolicy(pomdp::P, grid_size::Int=20, theta_grid_size::Int=4; 
                         n_episodes::Int=200000, learning_rate::Float64=0.1, 
                         epsilon::Float64=0.1) where P
        q_table = train_q_learning(pomdp, grid_size, theta_grid_size, 
                                 n_episodes, learning_rate, epsilon)
        new{P}(pomdp, grid_size, theta_grid_size, q_table)
    end
end

function train_q_learning(pomdp::MordorHikePOMDP, grid_size::Int, theta_grid_size::Int,
                         n_episodes::Int, learning_rate::Float64, epsilon::Float64)
    # Initialize Q-table with small random values for exploration
    q_table = rand(grid_size, grid_size, theta_grid_size, length(actions(pomdp))) * 0.01
    
    # Decay rates for learning rate and epsilon
    epsilon_decay = 0.995
    lr_decay = 0.995
    min_epsilon = 0.1
    min_lr = 0.1
    
    # Keep track of performance
    episode_rewards = Float64[]
    window_size = 100
    
    current_epsilon = epsilon
    current_lr = learning_rate
    
    for episode in 1:n_episodes
        s = rand(initialstate(pomdp))
        total_reward = 0.0
        step = 0
        max_steps = get_horizon(pomdp)
        
        while !isterminal(pomdp, s) && step < max_steps
            discrete_state = state_to_discrete(s, pomdp.map_upper_bound, 
                                            pomdp.map_lower_bound, 
                                            grid_size, theta_grid_size)
            
            # Epsilon-greedy with current epsilon
            if rand() < current_epsilon
                a = rand(actions(pomdp))
            else
                a = argmax(q_table[discrete_state..., :])
            end
            
            # Take action and observe next state and reward
            sp = rand(transition(pomdp, s, a))
            r = reward(pomdp, s, a, sp)
            total_reward += r
            
            next_discrete_state = state_to_discrete(sp, pomdp.map_upper_bound, 
                                                  pomdp.map_lower_bound, 
                                                  grid_size, theta_grid_size)
            
            # Q-learning update with current learning rate
            current_q = q_table[discrete_state..., a]
            next_max_q = maximum(q_table[next_discrete_state..., :])
            q_table[discrete_state..., a] = current_q + current_lr * 
                                          (r + pomdp.discount * next_max_q - current_q)
            
            s = sp
            step += 1
        end
        
        # Store episode reward
        push!(episode_rewards, total_reward)
        
        # Decay learning rate and epsilon
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        current_lr = max(min_lr, current_lr * lr_decay)
        
        # Print progress and running average
        if episode % 1000 == 0
            avg_reward = mean(episode_rewards[max(1, length(episode_rewards)-window_size+1):end])
            println("Episode $episode: Avg reward = $avg_reward, ε = $(round(current_epsilon, digits=3)), lr = $(round(current_lr, digits=3))")
        end
    end
    
    return q_table
end

function POMDPs.action(policy::QValuePolicy, s)
    # Convert state to discrete indices
    discrete_state = state_to_discrete(s, policy.pomdp.map_upper_bound, 
                                     policy.pomdp.map_lower_bound, 
                                     policy.grid_size, policy.theta_grid_size)
    
    # Return action with highest Q-value
    return argmax(policy.q_table[discrete_state..., :])
end

#play_interactive()

pomdp = MordorHikePOMDP(Val(:easy))
# Run a few Rollout policy and display paths
# println("\nRunning Rollout value iteration  policy demonstrations...")
#rollout_policy = GridValuePolicy(pomdp, 20, 4, 5000)
# display heatmap of value grid without Plots
# using GLMakie
# fig = Figure()
# ax = Axis(fig[1,1], title="Value Grid")
# for k in 1:rollout_policy.theta_grid_size
#     value_grid = rollout_policy.value_grid[:, :, k]
#     heatmap!(ax, value_grid, colormap=:viridis)
#     display(fig)
#     sleep(2.0)
# end

#rollout_policy = QValuePolicy(pomdp, 20, 4, n_episodes=200000)
# # display the heatmap of value grid computed from Q table for each angle
# # do argmax over actions
# value_grid = [maximum(rollout_policy.q_table[i, j, k, :]) for i in 1:rollout_policy.grid_size, j in 1:rollout_policy.grid_size, k in 1:rollout_policy.theta_grid_size]
# using GLMakie
# fig = Figure()
# ax = Axis(fig[1,1], title="Q Table")
# for k in 1:rollout_policy.theta_grid_size
#     v = value_grid[:, :, k]
#     heatmap!(ax, v, colormap=:viridis)
#     display(fig)
#     sleep(2.0)
# end

# rollout_policy = GoToGoalPolicy(pomdp)

# mdp = UnderlyingMDP(pomdp)
# hr = HistoryRecorder(max_steps=get_horizon(pomdp))

# # Run a few rollouts
# for i in 1:10
#     hist = simulate(hr, mdp, rollout_policy)
#     path = [s[1:2] for (s, b, a, r, sp, o) in hist]
#     last_state = hist[end][1]
#     display(render(pomdp, last_state, path=path))
#     sleep(1.0)
# end

# println("\nRunning POMCPOW with Value Iteration policy...")
# hr = HistoryRecorder(max_steps=get_horizon(pomdp))
# planner = solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(rollout_policy)), pomdp)
# for _ in 1:5
#     hist = simulate(hr, pomdp, planner)
#     # display the path and beliefs one by one
#     path = [h.s[1:2] for h in hist]
#     for i in 1:length(hist)
#         display(render(pomdp, hist[i].s, path=path[1:i], beliefs=hist[i].b.particles))
#         sleep(0.03)
#     end
#     sleep(1.0)
# end



# Define solvers
policy = Dict(
    # "POMCPOW (Random - deeper)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10000, max_depth=20), pomdp),
    #"POMCPOW (Random)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0)), pomdp),
    "POMCPOW (Value Iteration)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(GridValuePolicy(pomdp, 20, 20))), pomdp),
    # "POMCPOW (Value Iteration)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(GridValuePolicy(pomdp, 20, 4))), pomdp),
    # "POMCPOW (Q Learning 200k)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(QValuePolicy(pomdp, 20, 4, n_episodes=200000))), pomdp),
    # "POMCPOW (Q Learning 10k)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(QValuePolicy(pomdp, 20, 4, n_episodes=10000))), pomdp),
    "POMCPOW (GoToGoalPolicy)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(GoToGoalPolicy(pomdp))), pomdp),
    # "POMCPOW (GoToGoalPolicy) deeper" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(GoToGoalPolicy(pomdp)), tree_queries=10000, max_depth=20), pomdp),

    # "POMCPOW (Q Learning)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(QValuePolicy(pomdp, 20, 4))), pomdp),
    # "POMCPOW (A*)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(AStarRolloutPolicy(pomdp, 100))), pomdp),
    # "Value Iteration - Tree" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), estimate_value=FORollout(GridValuePolicy(pomdp, 100)), tree_queries=10000,max_depth=10), pomdp),

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
    success = []
    for i in 1:20
        hist = simulate(hr, pomdp, planner) 
        push!(returns, discounted_reward(hist))
        push!(scores, sum(h.r for h in hist))
        push!(lengths, length(hist))
        push!(success, isterminal(pomdp, hist[end].sp) ? 1.0 : 0.0)
        println("Return: $(returns[i]), Score: $(scores[i]), Length: $(lengths[i]), Success: $(success[i])")
        # display the path
        path = [h.s[1:2] for h in hist]
        last_state = hist[end].sp
        display(render(pomdp, last_state, path=path))
        sleep(1.0)
    end

    println(""" $policy_name Policy
        Mean and std of returns: $(mean(returns)), $(std(returns))
        Mean and std of scores: $(mean(scores)), $(std(scores))
        Mean and std of lengths: $(mean(lengths)), $(std(lengths))
        Mean and std of success: $(mean(success)), $(std(success))
        """)
end

# POMCPOW (A*) Policy
# Mean and std of returns: -35.16663408716645, 10.068539402802525
# Mean and std of scores: [-46.638096984262816], [16.373940241030596]
# Mean and std of lengths: 67.95, 18.782620746457695

# Medium

# POMCPOW (Random) Policy
# Mean and std of returns: -42.61298216910336, 10.344879274764946
# Mean and std of scores: [-66.30558364956326], [25.26126576533377]
# Mean and std of lengths: 97.3, 46.02527909518283

# POMCPOW (Value Iteration) Policy
# Mean and std of returns: -27.292398561554855, 8.409868977821864
# Mean and std of scores: [-34.1053362496223], [13.008224297066247]
# Mean and std of lengths: 54.85, 13.425329555888396


# Easy
# Mean and std of returns: -14.399205417350307, 4.553429429052851
# Mean and std of scores: [-16.41390277513299], [5.590275757963544]
# Mean and std of lengths: 35.85, 4.637093009520876
# Mean and std of success: 0.0, 0.0



# POMCPOW (GoToGoalPolicy) Policy
# Mean and std of returns: -31.29182244206683, 9.998530802807862
# Mean and std of scores: -40.60693033699148, 16.899584215662685
# Mean and std of lengths: 57.1, 19.226900015616827
# Mean and std of success: 1.0, 0.0

# POMCPOW (GoToGoalPolicy) deeper Policy
# Mean and std of returns: -30.06659092374474, 8.347961377413009
# Mean and std of scores: -39.17689921820302, 15.943191329133601
# Mean and std of lengths: 59.35, 24.781731386775945
# Mean and std of success: 1.0, 0.0

# POMCPOW (Q Learning 200k) Policy
# Mean and std of returns: -40.80231893659761, 11.57327411713214
# Mean and std of scores: -63.68373303281628, 28.7616261198471
# Mean and std of lengths: 100.05, 45.7366316390288
# Mean and std of success: 1.0, 0.0

# POMCPOW (Q Learning 10k) Policy
# Mean and std of returns: -46.72129276721404, 9.597723428027676
# Mean and std of scores: -78.9012706423197, 31.249547108573623
# Mean and std of lengths: 123.75, 54.655740778073806
# Mean and std of success: 1.0, 0.0

# POMCPOW (Value Iteration) Policy
# Mean and std of returns: -35.84783444633058, 13.616990981844936
# Mean and std of scores: -54.02855588351041, 28.768758220371815
# Mean and std of lengths: 85.4, 42.52974191514006
# Mean and std of success: 1.0, 0.0
# 0 init
# Mean and std of returns: -34.21073751036113, 10.130111066433408
# Mean and std of scores: -50.16481922781055, 22.666308143986377
# Mean and std of lengths: 83.95, 38.60525598249135
# Mean and std of success: 1.0, 0.0


# POMCPOW (GoToGoalPolicy) Policy
# Mean and std of returns: -25.911970239875064, 10.116077832608516
# Mean and std of scores: -32.470402455797846, 16.111342306915024
# Mean and std of lengths: 47.5, 19.674990804518682
# Mean and std of success: 1.0, 0.0


# POMCPOW (Value Iteration) Policy
# Mean and std of returns: -34.00081380403571, 11.330812582949031
# Mean and std of scores: -49.723270327329445, 26.42819918535351
# Mean and std of lengths: 81.4, 36.53318606706136
# Mean and std of success: 1.0, 0.0

# POMCPOW (Q Learning 200k) Policy
# Mean and std of returns: -47.82253752151657, 10.599107082095367
# Mean and std of scores: -81.82246837291237, 42.787461304106415
# Mean and std of lengths: 122.1, 71.68344152922405
# Mean and std of success: 0.95, 0.22360679774997902
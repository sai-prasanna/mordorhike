# import Pkg
# Pkg.add("POMDPs")
# Pkg.add("POMCPOW")
# Pkg.add("POMDPModels")
# Pkg.add("POMDPTools")
# Pkg.add("Distributions")
import POMDPs
import POMDPModels
import Distributions

using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using POMDPs
using POMDPTools
using LinearAlgebra
using Distributions
using Random
using Distributions: VonMises

@enum LateralAction strafe rotate



# Update the struct definition to include new fields
mutable struct MordorHikePOMDP <: POMDP{Vector{Float64}, Int64, Float64}
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
        0.0
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
    altitude = calculate_altitude(pomdp, sp[1:2])
    return Normal(altitude, pomdp.obs_std)
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

pomdp = MordorHikePOMDP(Val(:medium))
# Define solvers
rollout_policy = 
policy = Dict(
    "POMCPOW (deeper)" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10000, max_depth=30), pomdp),
    "POMCPOW" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10000, max_depth=10), pomdp),
    "POMCPOW - GoToGoalPolicy" => solve(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10000, max_depth=10, estimate_value=FORollout(GoToGoalPolicy(pomdp))), pomdp),
    "Random" => RandomPolicy(pomdp)
)

# Simulate
hr = HistoryRecorder(max_steps=get_horizon(pomdp))

# Run experiments for each solver
for (policy_name, planner) in policy
    
    println("\n$policy_name Policy")
    returns = []
    scores = []
    
    for i in 1:10
        hist = simulate(hr, pomdp, planner)
        push!(returns, discounted_reward(hist))
        push!(scores, sum(r for (s, b, a, r, sp, o) in hist))
        println("Return: $(returns[i]), Score: $(scores[i])")
    end

    println(""" $policy_name Policy
        Mean and std of returns: $(mean(returns)), $(std(returns))
        Mean and std of scores: $(mean(scores)), $(std(scores))
        """)
end

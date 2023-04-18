module SugarScape
# An implementation of the Epstein & Axtell (1996) model SugarScape. Original
# code from:
#
# https://juliadynamics.github.io/AgentsExampleZoo.jl/dev/examples/sugarscape/
#
# Very minor changes made by Fjalar de Haan in April 2023 to turn the tutorial
# code into a Julia module that can be loaded using `using SugarScape` and
# with the visualisations wrapped into functions for ease of demonstration.
#
# The website does not mention any licence but as Agents.jl is published under
# an MIT licence this package is under that licence too.
#
# Reference
# =========
#
# Epstein, J.M., Axtell, R., 1996. Growing Artificial Societies: Social Science
# from the Bottom up. Brookings Institution Press, Washington, D.C., USA.

export sugarscape, sugardash, sugarhist
export distances, sugar_caps
export agent_step!, model_step!
export move_and_collect!, replacement!

using Agents
using Random
using GLMakie, InteractiveDynamics

@agent SugarSeeker GridAgent{2} begin
    vision::Int
    metabolic_rate::Int
    age::Int
    max_age::Int
    wealth::Int
end

function distances(pos, sugar_peaks)
    all_dists = zeros(Int, length(sugar_peaks))
    for (ind, peak) in enumerate(sugar_peaks)
        d = round(Int, sqrt(sum((pos .- peak) .^ 2)))
        all_dists[ind] = d
    end
    return minimum(all_dists)
end

function sugar_caps(dims, sugar_peaks, max_sugar, dia = 4)
    sugar_capacities = zeros(Int, dims)
    for i in 1:dims[1], j in 1:dims[2]
        sugar_capacities[i, j] = distances((i, j), sugar_peaks)
    end
    for i in 1:dims[1]
        for j in 1:dims[2]
            sugar_capacities[i, j] = max(0, max_sugar - (sugar_capacities[i, j] ÷ dia))
        end
    end
    return sugar_capacities
end

"Create a sugarscape ABM"
function sugarscape(
                   ; dims = (50, 50)
                   , sugar_peaks = ((10, 40), (40, 10))
                   , growth_rate = 1
                   , N = 250
                   , w0_dist = (5, 25)
                   , metabolic_rate_dist = (1, 4)
                   , vision_dist = (1, 6)
                   , max_age_dist = (60, 100)
                   , max_sugar = 4
                   , seed = 42 )

    sugar_capacities = sugar_caps(dims, sugar_peaks, max_sugar, 6)
    sugar_values = deepcopy(sugar_capacities)
    space = GridSpaceSingle(dims)
    properties = Dict( :growth_rate => growth_rate
                     , :N => N
                     , :w0_dist => w0_dist
                     , :metabolic_rate_dist => metabolic_rate_dist
                     , :vision_dist => vision_dist
                     , :max_age_dist => max_age_dist
                     , :sugar_values => sugar_values
                     , :sugar_capacities => sugar_capacities )
    model = AgentBasedModel(
        SugarSeeker,
        space,
        scheduler = Schedulers.randomly,
        properties = properties,
        rng = MersenneTwister(seed)
    )
    for _ in 1:N
        add_agent_single!(
            model,
            rand(model.rng, vision_dist[1]:vision_dist[2]),
            rand(model.rng, metabolic_rate_dist[1]:metabolic_rate_dist[2]),
            0,
            rand(model.rng, max_age_dist[1]:max_age_dist[2]),
            rand(model.rng, w0_dist[1]:w0_dist[2]),
        )
    end
    return model
end

function model_step!(model)
    # At each position, sugar grows back at a rate of α units
    # per time-step up to the cell's capacity c.
    @inbounds for pos in positions(model)
        if model.sugar_values[pos...] < model.sugar_capacities[pos...]
            model.sugar_values[pos...] += model.growth_rate
        end
    end
    return
end

function agent_step!(agent, model)
    move_and_collect!(agent, model)
    replacement!(agent, model)
end

function move_and_collect!(agent, model)
    # Go through all unoccupied positions within vision, and consider the empty ones.
    # From those, identify the one with greatest amount of sugar, and go there!
    max_sugar_pos = agent.pos
    max_sugar = model.sugar_values[max_sugar_pos...]
    for pos in nearby_positions(agent, model, agent.vision)
        isempty(pos, model) || continue
        sugar = model.sugar_values[pos...]
        if sugar > max_sugar
            max_sugar = sugar
            max_sugar_pos = pos
        end
    end
    # Move to the max sugar position (which could be where we are already)
    move_agent!(agent, max_sugar_pos, model)
    # Collect the sugar there and update wealth (collected - consumed)
    agent.wealth += (model.sugar_values[max_sugar_pos...] - agent.metabolic_rate)
    model.sugar_values[max_sugar_pos...] = 0
    # age
    agent.age += 1
    return
end

function replacement!(agent, model)
    # If the agent's sugar wealth become zero or less, it dies
    if agent.wealth ≤ 0 || agent.age ≥ agent.max_age
        kill_agent!(agent, model)
        # Whenever an agent dies, a young one is added to a random empty position
        add_agent_single!(
            model,
            rand(model.rng, model.vision_dist[1]:model.vision_dist[2]),
            rand(model.rng, model.metabolic_rate_dist[1]:model.metabolic_rate_dist[2]),
            0,
            rand(model.rng, model.max_age_dist[1]:model.max_age_dist[2]),
            rand(model.rng, model.w0_dist[1]:model.w0_dist[2]),
        )
    end
end

function sugardash(model=nothing)
    if isnothing(model)
        model = sugarscape()
    end
    fig, ax, abmp = abmplot( model
                           ; agent_step!
                           , model_step!
                           , add_controls = true
                           , figkwargs = (resolution=(800, 600)) )
    # Lift model observable for heatmap
    sugar = @lift($(abmp.model).sugar_values)
    axhm, hm = heatmap(fig[1,2], sugar; colormap=:thermal, colorrange=(0,4))
    axhm.aspect = AxisAspect(1) # equal aspect ratio for heatmap
    Colorbar(fig[1, 3], hm, width = 15, tellheight=false)
    # Colorbar height = axis height
    rowsize!(fig.layout, 1, axhm.scene.px_area[].widths[2])
    display(fig)
    return fig, ax, abmp
end

function sugarhist(model=nothing, iterations=50)
    # AbstractPlotting.inline!(true)
    if isnothing(model)
        model = sugarscape()
    end
    adata, _ = run!( model
                   , agent_step!
                   , model_step!
                   , iterations
                   , adata = [:wealth] )
    figure = Figure(resolution = (600, 600))
    step_number = Observable(0)
    title_text = @lift("Wealth distribution of individuals, step = $($step_number)")
    Label(figure[1, 1], title_text; fontsize=20, tellwidth=false)
    ax = Axis(figure[2, 1]; xlabel="Wealth", ylabel="Number of agents")
    histdata = Observable(adata[adata.step .== 20, :wealth])
    hist!(ax, histdata; bar_position=:step)
    ylims!(ax, (0, 100))
    display(figure)
    for i in 0:iterations
        histdata[] = adata[adata.step .== i, :wealth]
        step_number[] = i
        xlims!(ax, (0, max(histdata[]...)))
        sleep(.1)
        display(figure)
    end

end


end # module SugarScape

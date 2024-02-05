using Plots
using GraphPlot
using TimerOutputs
using Statistics

include("./Filehandler/Filehandler.jl")
using .Filehandler
include("Route_eval.jl")
using .Route_eval
include("./Construction/Construction.jl")
using .Construction

using Revise



filename = raw"C:\codestuff\SanneWaste\Data\CARP benchmark\gdb12.ind"
currentinstance = read_arc_routing_file(filename)



const to = TimerOutput()

outputs = Dict()
for i in 1:100
    if(i<3)
        tours =augmentmerge(currentinstance)
       # verified = verify_augmentmerge(tours, currentinstance) #verify that all demand is covered
       # println("Verified that all edges are covered: ", verified)
    else
       @timeit to "augment_merge" tours=augmentmerge(currentinstance)
    
    end
    outputs[i] = augmentmerge_output(tours)
    #println( outputs[i] )
end
objvals = [value[1] for value in values(outputs)]
notours =[value[2] for value in values(outputs)]
average_obj= mean(objvals)
average_notour = mean(notours)
varience_obj= var(objvals)
println("average obj val is: $average_obj ; varience is: $varience_obj ; no of tours: $average_notour")
show(to)
# verified = verify_augmentmerge(tours, currentinstance)
#println("Verified that all edges are covered: ", verified)
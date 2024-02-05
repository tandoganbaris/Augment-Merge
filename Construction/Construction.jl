module Construction

using .Main.Filehandler
using .Main.Route_eval
using LightGraphs
using MetaGraphs:dijkstra_shortest_paths
import .Main.Filehandler:Node, EdgeCollection, Edge
using DataStructures

include("augment_merge.jl")


const vehicle_cap = Ref{Any}()
const distancematrix = Ref{Any}()
const depot =  Ref{Node}()
const alledges = Ref{EdgeCollection}()

function initialize_module(vehiclecap, dmatrix, Depotnode, insertededges)

    global vehicle_cap = vehiclecap
    global distancematrix = dmatrix
    global depot = Depotnode
    global alledges = insertededges
    println(
        "Module initialized with:: 
            capacity: $vehicle_cap , 
            depot: $(depot[].id), 
            this many nodes: $(length(distancematrix[1])), 
            this many edges: $(length(alledges))")
end


export initialize_module, augmentmerge, TourCARP, verify_augmentmerge, augmentmerge_output


end



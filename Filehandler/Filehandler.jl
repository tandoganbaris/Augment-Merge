
using LightGraphs

module Filehandler

using SimpleWeightedGraphs:SimpleWeightedGraph, add_edge!

using MetaGraphs
include("Fhmethods.jl")

export Instance, Node, Edge, EdgeCollection, read_arc_routing_file,create_graph,calculate_distance_matrix


end







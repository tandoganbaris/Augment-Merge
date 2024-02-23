using Graphs
using SimpleWeightedGraphs
using MetaGraphs #we are using dijkstra that takes simple weighted graph as input.
using GraphPlot
using SparseArrays


struct Node
    id::Int
   
    function Node(id::Int)::Node
        output= new(id)
        return output
    end
end


mutable struct Edge
    node1::Node
    node2::Node
    distance::Float64
    demand::Float64
    serviced::Bool
    # Add other parameters if needed

    function Edge(node1::Node, node2::Node, distance::Float64, demand::Float64, serviced::Bool =false)::Edge
        output= new(node1, node2, distance, demand,serviced)
        return output
    end
    function Edge()::Edge
        return new=(Node(1),Node(1), 0.0,0.0,false) 
    end

    Base.show(io::IO, edge::Edge)= print(io, "Edge ", edge.node1.id, " <-> ", edge.node2.id, " with dist/dem ", edge.distance ,"/", edge.demand)
end

    

mutable struct Instance
    alledges::Vector{Edge}
    allnodes::Vector{Node} 
    dmatrix::Any
    predmatrix::Any
    vehicle_cap::Int64
    adjacencylist::Dict{Node, Vector{Tuple{Node, Float64,Float64}}}#node to node and demand/distance
    graph::Any
    function Instance(alledges::Vector{Edge}, allnodes::Vector{Node} , dmatrix::Any,predmatrix::Any, vehicle_cap::Int64, adjacencylist::Dict{Node, Vector{Tuple{Node, Float64,Float64}}})::Instance
        return currentinstance= new(alledges, allnodes,dmatrix,predmatrix,vehicle_cap,adjacencylist,nothing)
    end

    
end
function read_arc_routing_file(filename)::Instance
    
    # Initialize variables to store node and edge information
    num_nodes = 0
    num_edges = 0
    edges = Vector{Edge}()
    nodes = Vector{Node}()
    adjacency_list = Dict{Node, Vector{Tuple{Node, Float64,Float64}}}()  # Dictionary to store adjacency list
    vehiclecap = 0
    # Open the file
    open(filename, "r") do file
        # Read the number of nodes
        num_nodes = parse(Int, readline(file))

        # Read the number of edges
        num_edges = parse(Int, readline(file))

        # Read edge information
        for _ in 1:num_edges
            line = readline(file)
            edge_data = split(line)
            
            # Parse edge information
            node1 = Node(parse(Int, edge_data[1])+1)
            node2 = Node(parse(Int, edge_data[2])+1)
            length = parse(Float64, edge_data[3])
            demand = parse(Float64, edge_data[4])
            
            # Check if node1 is already in the nodes vector
            if !(node1 in nodes)
                push!(nodes, node1)
            end

            # Check if node2 is already in the nodes vector
            if !(node2 in nodes)
                push!(nodes, node2)
            end

            # Store edge information
            if !(any(e -> ((e.node1 == node1 && e.node2 == node2) || (e.node1 == node2 && e.node2 == node1)), edges))
                # Add the edge only if both nodes are not already present in any edge
                push!(edges, Edge(node1, node2, length, demand))
                
            end
            push!(get!(adjacency_list, node1, Vector{Tuple{Node, Float64,Float64}}()), (node2, length,demand))
            push!(get!(adjacency_list, node2, Vector{Tuple{Node, Float64,Float64}}()), (node1, length,demand))
        end
        readline(file)
        vehiclecap = parse(Int, readline(file))
    
    
    
    end
    
    graph= create_graph(num_nodes, edges)
    matrix,predmatrix= calculate_distance_matrix(graph, num_nodes)
    
    currentinstance = Instance(edges, nodes, matrix,predmatrix, vehiclecap, adjacency_list)
    currentinstance.graph = graph
    return currentinstance
end
function read_raw_arc_routing_data(data::String)::Instance
    # Initialize variables to store node and edge information
    num_nodes = 0
    num_edges = 0
    edges = Vector{Edge}()
    nodes = Vector{Node}()
    adjacency_list = Dict{Node, Vector{Tuple{Node, Float64,Float64}}}()  # Dictionary to store adjacency list
    vehiclecap = 0
    # Open the file

     # Split the input string into lines
     lines = split(data, "\n")
    
     # Read the number of nodes
     num_nodes = parse(Int, lines[1])
 
     # Read the number of edges
     num_edges = parse(Int, lines[2])


    # Read edge information
    for i in 3:2+num_edges
        line = lines[i]
        edge_data = split(line)
        
        # Parse edge information
        node1 = Node(parse(Int, edge_data[1])+1)
        node2 = Node(parse(Int, edge_data[2])+1)
        length = parse(Float64, edge_data[3])
        demand = parse(Float64, edge_data[4])
        
        # Check if node1 is already in the nodes vector
        if !(node1 in nodes)
            push!(nodes, node1)
        end

        # Check if node2 is already in the nodes vector
        if !(node2 in nodes)
            push!(nodes, node2)
        end

        # Store edge information
        if !(any(e -> ((e.node1 == node1 && e.node2 == node2) || (e.node1 == node2 && e.node2 == node1)), edges))
            # Add the edge only if both nodes are not already present in any edge
            push!(edges, Edge(node1, node2, length, demand))
            
        end
        push!(get!(adjacency_list, node1, Vector{Tuple{Node, Float64,Float64}}()), (node2, length,demand))
        push!(get!(adjacency_list, node2, Vector{Tuple{Node, Float64,Float64}}()), (node1, length,demand))
    end
    vehiclecap = parse(Int, lines[num_edges + 4])

    
    
    
    
    graph= create_graph(num_nodes, edges)
    matrix,predmatrix= calculate_distance_matrix(graph, num_nodes)
    
    currentinstance = Instance(edges, nodes, matrix,predmatrix, vehiclecap, adjacency_list)
    currentinstance.graph = graph
    return currentinstance
end


function create_graph(num_nodes, edges)
    # Create a graph for adjacency representation
    graph = SimpleWeightedGraph(num_nodes)
    
    # Add edges to the graph
    for edge in edges
        SimpleWeightedGraphs.add_edge!(graph, edge.node1.id, edge.node2.id, edge.distance)
    end
    
    return graph
end


function calculate_distance_matrix(graph, num_nodes) #inefficient rn due to double for loop
    distance_matrix = zeros(Float64, num_nodes, num_nodes)
    predecessors_matrix =  [Int64[] for _ in 1:num_nodes, _ in 1:num_nodes]
    meta_graph = MetaGraph(graph)
    for i in 1:num_nodes
        shortest_result= MetaGraphs.dijkstra_shortest_paths(graph, i,allpaths=true)
        distance_matrix[i, :] = shortest_result.dists
        
    end
    for i in 1:num_nodes
        shortest_result= MetaGraphs.dijkstra_shortest_paths(graph, i,distance_matrix,allpaths=true)
        predecessors_matrix[i, :] = shortest_result.predecessors
    end
    
    return distance_matrix, predecessors_matrix
end
function get_weights_sparse(edges::Vector{Edge})
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    
    # Iterate through the edges and populate the arrays
    for edge in edges
        push!(rows, edge.node1.id)  # Row index (node1.id)
        push!(cols, edge.node2.id)  # Column index (node2.id)
        push!(vals, edge.distance)  # Value (distance)
    end
    
    # Create the sparse array
    sparse_array = sparse(rows, cols, vals)
    dropzeros!(sparse_array)
    return sparse_array
end


#filename = raw"C:\codestuff\SanneWaste\Data\CARP benchmark\gdb12.ind"
#currentinstance = read_arc_routing_file(filename)

data ="5
8
0  4 15 1
0  1 7 1
0  3 10 1
1  2 6 1
1  4 8 1
2  4 4 1
2  3 9 1
3  4 6 1
  8
  5
0
50"
currentinstance = read_raw_arc_routing_data(data)

#PLOTTING FOR VISUALIZATION
weights_dict = nonzeros(get_weights_sparse(currentinstance.alledges))
gplot(currentinstance.graph, nodefillc="white", edgestrokec="black",
     nodelabel=[string(i) for i in 1:length(currentinstance.allnodes)],
     edgelabel=weights_dict,edgelabelc="white",
     layout=circular_layout)


#DEMO OF DIJKSTRA
graph = MetaGraph(currentinstance.graph)
startnode = 3 #node/vertex to calculate distances and predecessors from 
result= Graphs.dijkstra_shortest_paths(graph, startnode ,currentinstance.dmatrix,allpaths=true)
distances = result.dists
preds = result.predecessors
println("Distances between input node $startnode to all other nodes: $distances \n")
println("Predecessors from node $startnode to other nodes: $preds \n BEWARE: This matrix does not show the shortest predecessors without weights!")
println("Predecesor matrix will have nothing if startnode=endnode and it will have startnode if endnode is directly connected to start node, e.g. if 1 is connected to 2 the matrix will have '[1]' at index 2")

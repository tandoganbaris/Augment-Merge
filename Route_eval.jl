 


module Route_eval
include("./Filehandler/Filehandler.jl")
#using .Filehandler
using Random 
#FLIPPY GRAPH
# distance matrix between nodes, edges of the tour that must be traversed in an array, depot node
function RoutecostCARP(distance_matrix, edge, depot)
    edgeinternal = edge[:]
    
    if (edge[1].node1.id == 1 || edge[1].node2.id==1)
        edgeinternal = edgeinternal[2:end]
    end
    if (edge[end].node1.id == 1 || edge[end].node2.id==1)
        edgeinternal = edgeinternal[1:end-1]
    end
    
    costmatrix = fill(Inf,2, length(edgeinternal))

    for i in 1:(size(costmatrix,2)) #range inclusive 
        currentedge = edgeinternal[i]
        if(i-1>0) #there exists an edge before
            prevedge = edgeinternal[i-1] 
            

              #costmatrix row up is to arrive next upper node min dist
            costmatrix[1,i]= min((costmatrix[1,i-1]+ distance_matrix[prevedge.node1.id, currentedge.node1.id]),
                        (costmatrix[2,i-1]+ distance_matrix[prevedge.node2.id, currentedge.node1.id]))

      
            #costmatrix row down is to arrive next lower node min dist
            costmatrix[2,i]= min((costmatrix[1,i-1]+ distance_matrix[prevedge.node1.id, currentedge.node2.id]),
                        (costmatrix[2,i-1]+ distance_matrix[prevedge.node2.id, currentedge.node2.id]))
        elseif i-1==0 #previously was depot
            costmatrix[1,i]= distance_matrix[depot.id, currentedge.node1.id]
            costmatrix[2,i]= distance_matrix[depot.id, currentedge.node2.id] 

        end
    end
    last_column = costmatrix[:,end]
    routecost = min(last_column[1]+ distance_matrix[depot.id, edgeinternal[end].node1.id],
                last_column[2]+ distance_matrix[depot.id, edgeinternal[end].node2.id] )
    routecost = routecost + sum(edge.distance for edge in edges)
    return routecost

end

function RoutecostCARP2(distance_matrix, edge_collection::Filehandler.EdgeCollection, depot)
    edgeinternal = edge_collection.edges[:]
    #=

    if (edge_collection.edges[1].node1.id == 1 || edge_collection.edges[1].node2.id == 1)
        edgeinternal = edgeinternal[2:end]
    end
    if (edge_collection.edges[end].node1.id == 1 || edge_collection.edges[end].node2.id == 1)
        edgeinternal = edgeinternal[1:end-1]
    end
    =#
    costmatrix = fill(Inf, 2, length(edgeinternal))

    for i in 1:(size(costmatrix, 2))
        currentedge = edgeinternal[i]
        if (i - 1 > 0)
            prevedge = edgeinternal[i - 1]
            
            # Update cost matrix
            costmatrix[1, i] = min((costmatrix[1, i - 1] + distance_matrix[prevedge.node1.id, currentedge.node1.id]),
                                   (costmatrix[2, i - 1] + distance_matrix[prevedge.node2.id, currentedge.node1.id]))
            
            costmatrix[2, i] = min((costmatrix[1, i - 1] + distance_matrix[prevedge.node1.id, currentedge.node2.id]),
                                   (costmatrix[2, i - 1] + distance_matrix[prevedge.node2.id, currentedge.node2.id]))
        elseif i - 1 == 0
            costmatrix[1, i] = distance_matrix[depot.id, currentedge.node1.id]
            costmatrix[2, i] = distance_matrix[depot.id, currentedge.node2.id]
        end
    end

    last_column = costmatrix[:, end]
    routecost = min(last_column[1] + distance_matrix[depot.id, edgeinternal[end].node1.id],
                    last_column[2] + distance_matrix[depot.id, edgeinternal[end].node2.id])
    routecost = routecost + sum(edge.distance for edge in edge_collection.edges)
    return routecost
end
end

# Function to create a symmetric random matrix
#=
function symmetric_random_matrix(size)
    # Initialize a matrix with zeros
    mat = zeros(size, size)
    
    # Fill the upper triangle with random values
    for i in 1:size
        for j = i+1:size
            mat[i, j] = rand(1:100) # random distance between 1 and 100
        end
    end
    
    # Make the matrix symmetric
    mat = mat + transpose(mat)
    
    return mat
end
end

num_node =10 
distmattest= symmetric_random_matrix(num_node)
# Initialize a list to store selected edges
selected_edges = []

# Generate a random permutation of nodes (excluding the first node)
nodes_permutation = shuffle(2:num_node)



# Loop to select random edges
push!(selected_edges, Edge(Node(1), Node(nodes_permutation[1]), distmattest[1,nodes_permutation[1]],0.0))
for i in 1:(num_node-2)
    # Get two consecutive nodes from the permutation
    node1 = nodes_permutation[i]
    node2 = nodes_permutation[i + 1]
    
    # Add the edge to the selected edges list
    push!(selected_edges, Edge(Node(node1), Node(node2), distmattest[node1,node2],0.0))
end

# Add an edge to connect the last node back to the first node
push!(selected_edges, Edge(Node(1), Node(nodes_permutation[end]), distmattest[1,nodes_permutation[end]],0.0))
#=
# Print the selected edges
println("Selected Edges:")
println(selected_edges)


costtest = RoutecostCARP(distmattest, selected_edges, Node(1))
print("cost of route is: " , costtest )
=#
selected_edge_collection = EdgeCollection(selected_edges)

# Print the selected edges
println("Selected Edges:")
for edge in selected_edge_collection.edges
    println(edge)
end

# Calculate and print the cost of the route
costtest = RoutecostCARP2(distmattest, selected_edge_collection, Node(1))
print("Cost of route is: ", costtest)
=#
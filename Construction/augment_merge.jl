
mutable struct TourCARP
    depot::Node
    edges::Vector{Edge}
    servicededges::Vector{Edge}
    totaldistance::Float64
    enteraugmentmerge::Tuple{Bool,Bool} #enter augment, enter merge
    path::Vector{Int}
    demand::Int64
    
    function TourCARP(depot::Node, edges::Vector{Edge}, servicededges::Vector{Edge},  totaldistance::Float64, enteraugmentmerge::Tuple{Bool,Bool} = (true,true), path::Vector{Int} = [])::TourCARP
        demand =0 
        if(length(servicededges)!= 0 )
            demand= sum(edge.demand for edge in servicededges)
        end
         
        
        return new(depot, edges,servicededges, totaldistance, enteraugmentmerge, path, demand)
    end
    
    function TourCARP()::TourCARP
        return new(Node(1),Vector{Edge}(),Vector{Edge}(),0.0,(true,true), Vector{Int64}(),0)
        
    end
    


end
function totaldemand(tour::TourCARP)
    if(length(tour.servicededges)!= 0 )
        tour.demand= sum(edge.demand for edge in tour.servicededges)
    else
        tour.demand=0.0
    end
    return tour.demand
end


function reversetour(tour::TourCARP)
    tour.edges = reverse(tour.edges)
    tour.path = reverse(tour.path)
end
function hash_tour(tour::TourCARP)
    h = hash(:TourCARP)
    for edge in tour.edges
        h = hash(edge, h)
    end
    for serviced_edge in tour.servicededges
        h = hash(serviced_edge, h)
    end
    return h
end


function augmentmerge(instance)
    vehicle_cap = instance.vehicle_cap
    distancematrix = instance.dmatrix
    depot =  instance.allnodes[1]
    alledges = instance.alledges
    tours = initialize_tours(depot, alledges,instance)
    
    augmentedtours = augment(tours , vehicle_cap)
    mergedtours = merge(augmentedtours , vehicle_cap, distancematrix, instance)
    return mergedtours
end

function initialize_tours(depot, edge_collection,instance)
    tours= TourCARP[] 
    #for all edges that have positive demand construct depot-i-j-depot
    for i in 1:(length(edge_collection))
        if(edge_collection[i].demand>0) #only make tour if theres demand. 
            target_edge = edge_collection[i]
            target_edge.serviced = true

            
            predecessors1,edges1,distances1 = reconstruct_path(instance.predmatrix,depot,target_edge.node1,edge_collection)
            predecessors2, edges2, distances2 = reconstruct_path(instance.predmatrix,depot,target_edge.node2,edge_collection)

            touredges = Vector{Edge}()
            remove_edge!(edges1, target_edge) #check if the path going to i or j has i,j in it, and delete if so.
            remove_edge!(edges2, target_edge)
            if(length(edges1)>0)             
                append!(touredges, reverse(edges1))
            end
            if(is_edge_in_collection(target_edge,(edges1))[1]==false)&&
                (is_edge_in_collection(target_edge,(edges2))[1]==false)
                 
                push!(touredges, target_edge)
            end #only push if paths dont contain target edge
            if(length(edges2)>0)
                append!(touredges, edges2)
            end           
                       
            totaldist = distances1 + target_edge.distance + distances2
                                            
            pathcomb = vcat(reverse(predecessors1),predecessors2)

            #touredges = EdgeCollection(touredges)
            servicededges = Vector{Edge}() 
            push!(servicededges, target_edge)
            

            newtour= TourCARP(depot,touredges,servicededges,totaldist,(true,true), pathcomb)
            
          
            push!(tours, newtour)
            
        end
    end
    return tours
end


function augment(tours, vehicle_cap::Int64)
    # Sort tours by distance in descending order
    sorted_tours = sort!(tours, by = tour -> tour.totaldistance, rev = true)
    full_tours= []
    # Iterate over the sorted tours
    for i in eachindex(deepcopy(sorted_tours))
        current_tour = sorted_tours[i]
        if(current_tour.enteraugmentmerge[1]==false)
            continue
        end

        # Check against all other tours
        for j in i+1:length(deepcopy(sorted_tours))
            smaller_tour = sorted_tours[j]
            if(smaller_tour.enteraugmentmerge[1]==false)
                continue
            end
             # Iterate over edges in the smaller tour
            # Check if the smaller tour shares multiple edges with the current tour
            shares, sharededges =shares_multiple_edges(current_tour, smaller_tour)
            if shares
            # Iterate over edges in the smaller tour
                for edge in sharededges
                 # Check if the arc can be added to the current tour
                    if can_be_serviced(current_tour, edge, vehicle_cap)
                    # Service the edges on the current tour
                        addtest = add_edge(current_tour, edge)
                        if(!addtest)
                            return false                    
                        end
                        removeedge(smaller_tour, edge)
                    end
                    # If smaller_tour has no more edges, mark it as discarded
                    
                    if isempty(smaller_tour)
                        smaller_tour.enteraugmentmerge=(false, smaller_tour.enteraugmentmerge[2]) #eliminate smaller tour
                        break  # Break out of the edge loop
                    end
                end
            end
           
        

            # If the current tour reaches capacity, set it aside as completed
            if reaches_capacity(current_tour, sorted_tours, vehicle_cap)
                break  # Break out of the smaller tour loop
            end
        
        end
        push!(full_tours, deepcopy(current_tour)) #after for this tour all other combinations are checked
        current_tour.enteraugmentmerge=(false,current_tour.enteraugmentmerge[2]) #remove the full tour
        

    end

    # Return the modified tours
    return full_tours # not really full but cannot be augmented anymore
end
function merge(tours, vehiclecap, distancematrix, instance)
    savingsdict = createsavings(tours, vehiclecap, distancematrix, instance)

    while length(keys(savingsdict)) > 0
        key, val = first(savingsdict)
        chosentour = val[2]
        tours, savingsdict = updatesavings(savingsdict, key, tours, chosentour, instance)
        
    end
    return tours
end
function updatesavings(savingsdict, chosenkey, tours, chosentour, instance) 
    tours = filter(tour -> !(hash_tour(tour) in chosenkey), tours) #remove used tours
    
    if !reaches_capacity(chosentour,tours, instance.vehicle_cap)
        push!(tours, chosentour) # add chosentour to tours
         
        for key in keys(savingsdict)
            if any(x -> x in key, chosenkey[1]) || any(x -> x in key, chosenkey[2])
                delete!(savingsdict, key)
            end
        end
        savingsdict = createsavings_fornewtour(tours, savingsdict, chosentour, instance) #with chosen tour and all other tours
    else
        chosentour.enteraugmentmerge = (false,false)
        push!(tours, chosentour) #add chosentour
        for key in keys(savingsdict)
            if any(x -> x in key, chosenkey[1]) || any(x -> x in key, chosenkey[2])
                delete!(savingsdict, key)
            end
        end 

    end
    return tours, savingsdict
end
function createsavings_fornewtour(tours, savingsdict, chosentour, instance)#todo
    notfulltours = filter(t -> t.enteraugmentmerge[2]==true, tours)
        for (j, tour2) in enumerate(notfulltours)
            if hash_tour(chosentour) != hash_tour(tour2) # Avoid comparing a tour with itself
                proceedmerge = ((chosentour.demand + tour2.demand) < instance.vehicle_cap)
                if proceedmerge
                     savings, mergedtour = calculate_savings(chosentour, tour2, instance.dmatrix, instance)
                     key = (hash_tour(chosentour), hash_tour(tour2))
                     savingsdict[key] = (savings, mergedtour)     
                else
                    continue
                end
            end
        end
    sorted_pairs = sort([(k, v) for (k, v) in savingsdict], by=x -> x[2][1], rev=true) #val== tuple(saving,mergedtour)
    savingsdict = DataStructures.OrderedDict(sorted_pairs)
    return savingsdict
end
function createsavings(tours, vehiclecap, dmatrix, instance)
    savings_dict = Dict()
    notfulltours = filter(t -> t.enteraugmentmerge[2]==true, tours)
    for (i, tour1) in enumerate(notfulltours)
        for (j, tour2) in enumerate(notfulltours)
            if ((i != j) && ((tours[i].demand + tours[j].demand) <vehiclecap)) # Avoid comparing a tour with itself
                            
                saving, mergedtour = calculate_savings(tours[i], tours[j],dmatrix, instance)
                if(saving>0)
                    key = (hash_tour(tours[i]), hash_tour(tours[j]))
                    savings_dict[key] = (saving, mergedtour)
                end
            end
        end
    end
    sorted_pairs = sort([(k, v) for (k, v) in savings_dict], by=x -> x[2][1], rev=true) #val== tuple(saving,mergedtour)
    savingsdict = DataStructures.OrderedDict(sorted_pairs)
    return savingsdict
end
function merge_external_edges(edges1, edges2,dmatrix) #add subsections of edges together to create tour
    returnnode = edges1[end]
    restartnode = edges2[1]
    saving = dmatrix[returnnode, 1] + dmatrix[1, restartnode] - dmatrix[returnnode, restartnode]
    return saving, (edges1, edges2)
end
function calculate_savings(tour1,tour2,dmatrix, instance)
    t1_firstserve = findfirst(e -> e.serviced==true, tour1.edges)
    t1_lastserve = findlast(e -> e.serviced==true, tour1.edges)
    tour1pathindex1 = find_first_node_match(tour1.edges[t1_firstserve], tour1.path)
    tour1pathindex2 = find_last_node_match(tour1.edges[t1_lastserve], tour1.path)
    subpath1 = deepcopy(tour1.path[tour1pathindex1:tour1pathindex2])
    #take tour1 and tour2 subsections of service
    t2_firstserve = findfirst(e -> e.serviced==true, tour2.edges)
    t2_lastserve = findlast(e -> e.serviced==true, tour2.edges)
    tour2pathindex1 = find_first_node_match(tour2.edges[t2_firstserve], tour2.path)
    tour2pathindex2 = find_last_node_match(tour2.edges[t2_lastserve], tour2.path)
    subpath2 = deepcopy(tour2.path[tour2pathindex1:tour2pathindex2])
    

    #Merge1 as is
    saving1, merged1 = merge_external_edges(subpath1, subpath2,dmatrix)
    #Merge2 reverse order
    saving2, merged2 = merge_external_edges(reverse(subpath1), reverse(subpath2), dmatrix)
    #Merge3 t1 reverse
    saving3, merged3 = merge_external_edges(reverse(subpath1), subpath2, dmatrix)
    #Merge4 t2 reverse
    saving4, merged4 = merge_external_edges(subpath1, reverse(subpath2), dmatrix)

    savings=[saving1,saving2,saving3, saving4]
    mergedpaths = [merged1, merged2, merged3, merged4]
    best_saving_index = argmax(savings)
    best_path_tuple= mergedpaths[best_saving_index]
    tour1indices = (t1_firstserve, t1_lastserve, tour1pathindex1, tour1pathindex2)
    tour2indices = (t2_firstserve, t2_lastserve, tour2pathindex1, tour2pathindex2)
    mergedtour = reconstruct_tour(tour1, tour2, best_path_tuple, tour1indices, tour2indices,best_saving_index, instance)
    return savings[best_saving_index], mergedtour
end
#=
reconstruct from parts of tours with service. the indices are first for edges (begin end service) and then for path (begin end) tuple(e_begin, e_end, p_begin, p_end)
savingsindex gives which style of merge was best_path_tuple
instance is input in order to have predecessors for path construction. 
=#
function reconstruct_tour(tour1::TourCARP, tour2::TourCARP, path_tuple, t1indices, t2indices , savingsindex, instance )
    mergedtour = TourCARP()
    edgesmerged = Vector{Filehandler.Edge}()
    path = Vector{Int64}()
    if(savingsindex==1)
        append!(path, tour1.path[1: t1indices[4]]) #take path up to end of demand
        append!(edgesmerged, tour1.edges[1: t1indices[2]])
        midpath, midegdes, middist = reconstruct_path(instance.predmatrix, path_tuple[1][end], 
        path_tuple[2][1], instance.alledges) 
        append!(path, midpath)#add midsection to path
        append!(edgesmerged,midegdes)
        append!(edgesmerged, tour2.edges[t2indices[1]:end])
        append!(path, tour2.path[t2indices[3]:end]) # take path from beginning of demand till depot
    elseif(savingsindex==2) #both paths reversed
        append!(path,  reverse(tour1.path[t1indices[3]:end])) 
        append!(edgesmerged, reverse(tour1.edges[t1indices[1]:end ]))
        midpath, midegdes, middist = reconstruct_path(instance.predmatrix, path_tuple[1][end], 
        path_tuple[2][1], instance.alledges) 
        append!(path, midpath)#add midsection to path
        append!(edgesmerged,midegdes)
        append!(edgesmerged, reverse(tour2.edges[1:t2indices[2]]))
        append!(path,  reverse(tour2.path[1:t2indices[4]])) 
    elseif(savingsindex==3)
        append!(path,  reverse(tour1.path[t1indices[3]:end])) 
        append!(edgesmerged, reverse(tour1.edges[t1indices[1]:end ]))
        midpath, midegdes, middist = reconstruct_path(instance.predmatrix, path_tuple[1][end], 
        path_tuple[2][1], instance.alledges) 
        append!(path, midpath)#add midsection to path
        append!(edgesmerged,midegdes)
        append!(edgesmerged, tour2.edges[t2indices[1]:end])
        append!(path, tour2.path[t2indices[3]:end]) # take path from beginning of demand till depot
    elseif(savingsindex==4)
        append!(path, tour1.path[1: t1indices[4]]) #take path up to end of demand
        append!(edgesmerged, tour1.edges[1: t1indices[2]])
        midpath, midegdes, middist = reconstruct_path(instance.predmatrix, path_tuple[1][end], 
        path_tuple[2][1], instance.alledges) 
        append!(path, midpath)#add midsection to path
        append!(edgesmerged,midegdes)
        append!(edgesmerged, reverse(tour2.edges[1:t2indices[2]]))
        append!(path,  reverse(tour2.path[1:t2indices[4]])) 

    end
    mergedtour.edges = edgesmerged
    mergedtour.path = path
    mergedtour.depot = tour1.depot
    mergedtour.enteraugmentmerge = tour1.enteraugmentmerge
    for edge in mergedtour.edges
        mergedtour.totaldistance += edge.distance
        if edge.serviced
            push!(mergedtour.servicededges, edge)
            mergedtour.demand += edge.demand
        end
    end
    return mergedtour

end
function find_first_node_match(edge::Edge, path::Vector{Int64})
    for (i, node_id) in enumerate(path)
        if node_id == edge.node1.id || node_id == edge.node2.id
            return i  # Return the index of the first matching node
        end
    end
    return nothing  # Return nothing if no match is found
end
function find_last_node_match(edge::Edge, path::Vector{Int64})
    reversed_index = 0
    for (index, node_id) in enumerate(reverse(path))
        if index < length(path) && path[length(path) - index] == edge.node1.id || path[length(path) - index] == edge.node2.id
            reversed_index = index
            break
        end
    end
    og_index = length(path) - reversed_index 
    if(og_index != 0 )
        return og_index
    end

    return nothing  # Return nothing if no match is found
end

function shares_multiple_edges(current_tour, smaller_tour)
    # Create a set of edges for each tour for efficient lookup
    edges_tourc = current_tour.edges
    edges_tours = smaller_tour.servicededges
    minoverlap = 1
    overlap = 0
    shares = false
    sharededges = Vector{Filehandler.Edge}()
    for edge in edges_tours #iterate servicededges of smaller tour
        largertour_traverses, foundedge = is_edge_in_collection(edge, edges_tourc) #if larger tour has these edges traversed
        if largertour_traverses
            isrecorded, foundedge2 = is_edge_in_collection(edge, sharededges)#shared edges doesnt have this edge yet 
            if isrecorded==false
                push!(sharededges, deepcopy(edge))
                shares = true   
            end               
        end
    end
    return shares, sharededges 
end
function isempty(tour::TourCARP)::Bool
    demand = totaldemand(tour)
    return demand == 0.0
end
# Helper functions (to be implemented based on your data structure)
function can_be_serviced(tour::TourCARP, edge, vehicle_cap::Int64)::Bool
    return tour.demand+edge.demand <= vehicle_cap
end


function add_edge(tour, edge)
    indices = Int[]
    indices = findall(e -> compare_edges(e,edge), tour.edges)
    #append!(indices, findall(e -> e== Filehandler.Edge(edge.node2, edge.node1, edge.distance, edge.demand), tour.edges))
    output = true
    if(length(indices)>1) #if edge is traversed multiple times, we service last to support merge
        tour.edges[last(indices)].serviced=true
        
        
    elseif(length(indices)==1)
        tour.edges[indices[1]].serviced=true
        

    else 
        output =false
    end
    tour.servicededges = Vector{Edge}()
    for edge in tour.edges
        if edge.serviced==true
            push!(tour.servicededges, edge)
        end
    end

   
    # Add the arc to the tour
    return output
end

function reaches_capacity(tour::TourCARP, tours, vehicle_cap::Int64)::Bool
    # Check if the tour has reached its capacity
    existing = totaldemand(tour)
    lowdemandtour= argmin(t -> totaldemand(t), tours)
    if((existing + lowdemandtour.demand) > vehicle_cap)
        tour.enteraugmentmerge = (false,false) 
        return true
    else 
        return false
    end
end
# Assuming 'g' is your MetaGraph and 'depot' is your starting node
#distances, predecessors = MetaGraphs.dijkstra_shortest_paths(g, depot)

function reconstruct_path(predecessors, start_node::Node, end_node::Node,edgecollection::Vector{Edge})
    path = [start_node.id]
    edges = Vector{Filehandler.Edge}()
    distance = 0 
    current = start_node.id
    while current != end_node.id
        pred = predecessors[end_node.id,current]
        if pred[1]!=current #dijkstra returns same id if current connects to start
            append!(path, pred)
            exists, foundedge = is_edge_in_collection(Filehandler.Edge(Node(pred[1]), Node(current),0.0,0.0), edgecollection)
            if exists
                foundedge = deepcopy(foundedge)
                foundedge.serviced=false
                push!(edges, foundedge) 
            end 
            current = pred[1]
        else
            #exists, foundedge = is_edge_in_collection(Filehandler.Edge(Node(start_node.id), deepcopy(current),0.0,0.0), edgecollection)
            #if exists
            #    push!(edges, foundedge)
            #end 
            current =end_node.id
        end
    end
    if(length(edges)>0) 
        for edge in edges
            distance += edge.distance
        end
    end
    reverse!(path)  # Reverse to get the correct order
    reverse!(edges) 
    return path, edges,distance
end

function reconstruct_path(predecessors, start_node::Int64, end_node::Int64,edgecollection::Vector{Edge})
    path = [start_node]
    edges = Vector{Filehandler.Edge}()
    distance = 0 
    current = start_node
    while current != end_node
        pred = predecessors[end_node,current]
        if pred[1]!=current #dijkstra returns same id if current connects to start
            push!(path, pred[1])
            exists, foundedge = is_edge_in_collection(Filehandler.Edge(Node(pred[1]), Node(current),0.0,0.0), edgecollection)
            if exists
                foundedge = deepcopy(foundedge)
                foundedge.serviced=false
                push!(edges, deepcopy(foundedge)) 
            end 
            current = pred[1]
        else
            #exists, foundedge = is_edge_in_collection(Filehandler.Edge(Node(start_node.id), deepcopy(current),0.0,0.0), edgecollection)
            #if exists
            #    push!(edges, foundedge)
            #end 
            current =end_node
        end
    end
    if(length(edges)>0) 
        for edge in edges
            distance += edge.distance
        end
    end
    reverse!(path)  # Reverse to get the correct order
    reverse!(edges) 
    return path, edges,distance
end

function removeedge(tour::TourCARP, edge)
    indices = findall(e -> compare_edges(e,edge), tour.edges)
    for index in indices
        tour.edges[index].serviced = false
    remove_edge!(tour.servicededges, edge)
    end

end

#removed an edge from a path to avoid duplicates
function remove_edge!(edges::Vector{Filehandler.Edge}, target_edge::Filehandler.Edge)::Bool
    indices_to_remove = findall(edge -> compare_edges(edge, target_edge), edges)
    if length(indices_to_remove)>0
        for index_to_remove in indices_to_remove
            splice!(edges, index_to_remove)
            return true
        end
    else
        return false
    end
end

function compare_edges(edge1::Filehandler.Edge, edge2::Filehandler.Edge)::Bool
    return ((edge1.node1 == edge2.node1 && edge1.node2 == edge2.node2) ||
           (edge1.node1 == edge2.node2 && edge1.node2 == edge2.node1))
end

function is_edge_in_collection(edge::Filehandler.Edge, edges_collection::Vector{Filehandler.Edge})
    foundedge = nothing
    for e in edges_collection
        if compare_edges(edge, e)
            foundedge = deepcopy(e)
            return true , foundedge
        end
    end
    return false , nothing
end

function verify_augmentmerge(tours, instance)::Bool
    positive_demand_edges = Vector{Edge}()

    for edge in instance.alledges
        if edge.demand > 0
            push!(positive_demand_edges, edge)
        end
    end
    covered_edges= Vector{Edge}()
    for tour in tours
        for edge in tour.servicededges
            push!(covered_edges, edge)
        end
    end
    
    covered_all=true
    for edge_to_remove in covered_edges
        covered_all, foundedge = is_edge_in_collection(edge_to_remove, positive_demand_edges)
        if covered_all== false
            break
        end
    end
    return covered_all

end

function augmentmerge_output(tours)
    objval =  0
    for tour in tours
        objval +=tour.totaldistance
    end

    return (objval, length(tours))
    
end
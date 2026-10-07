there are several many different possible designs.

genotype space:
- restrict genotypes to be DAGs
- do not restrict genotypes to be DAGs. the genotypes can be circular and phenotypes can be DAGs. the extent how circular the graph is can be an optimization target


for graph genotype with DAG phenotypes, consider
- how to map a graph to valid DAG (decoder)
 + find a subtree, with largest weight of edges. then how to design the weight of edges(if we already know IBD values)? in order to limit the cost of building DAG, the circular extent should be restricted i.e. individual genotypes with circular metric > threshold should not be kept
 + other methods

- how to measure circular extent
 + number of edges to remove to be DAG 
 + other 

- if weight of edge is used, how to measure 
 

genotype representation:
- use NEAT-like ways, and with directions distinguished. 
 + mutation: start with simple structures, then add individuals and relations, each change will have an innovation number to track history. 

 + crossover: align with innovation number before crossover 
- use CGP(cartesian genetic programming)-like ways,  reserve 2 inputs for each individual. 
- tree GP: each individual is a tree how the pedigree is generated, instead of being pedigree itself.
```
    possible structures:
    AddParents(
        AddSibling(
            ExtendAncestor(A)
        )
    )
    or 
    Family(
        parents = Pair(X1, X2),
        children = [
            A,
            AddSibling(B)
        ]
    )
```
 + crossover: select a subtree and interchange
 

possible mutations in general:
    add latent parent
    add/remove parent-child edge 
    activate/deactivate latent person
    split one ancestor into two
    merge two latent ancestors
    attach a person to an existing parent pair
    detach and reattach a small branch
    insert individual to reduce relationship degree
small mutations: remove x1 -> A and add x3 -> A 
medium mutations: 
 - A and B have different parent pairs -> same parent pair 
   or add latent parent for sibling groups 
major mutations:
 - merge two ancestors, split one ancestor, move a whole branch, replace one common ancestor structure



should we consider mutation strength in the program explicitly?
how to calculate genotypic mutation size ?
usage of mutation size
 - for possibility whether a candidate mutation is applied
 - for self adapted mutation strength: let each individual carry a number sigma that can restrict acceptable strength.  
 - to accept different strength as the optimization goes.
 - for adaptive operator selection: if one operator makes more improvement, increase the possibility it would be selected


objective designs:

I hope to use multiobjectives. But it needs to consider how to design these 
objectives. For example, how well the candidate solution fits the given IBD
can be either one (abs or squared) summed error, or multiple different objectives,
including the total error as well as numbers of pairs of people that have IBD error 
\>= different thresholds, or it can be mean error as well as worst error.

rationale:
if there are 20 pairs of people, 19 are fit and 1 is completely wrong, the mear 
error can be well but there is 1 severe error that indicates that the topology 
has very fundamental problem.

>draft:
candidate metrics:
> - IBD overall error
> - worst-pair error 
> - latent complexity(whether unnecessary structures introduced, while there can be simpler structures, and do not affect the relations between explicit individuals)
> - structural validity
> - efficiency to explain: how much observed relationship can be explained (improved) by adding a new latent person; IBD fit improvement/number of latent structures



other metrics: to discuss in future.


how to treat latent parents 
latent parents should also be encoded. 
discuss: whether all complete solutions should have all non root people have both parents assigned. 
should we leave it open for intermediate solutions, and compute all possible IBD for different missing parent assignments?


----

# Program and project design
we start from simple basic designs and add features later. 
But there are different designs with completely different structures. 
So we need to figure out several basic designs to start with. 

different python packages may be used for different designs
but I wonder whether there are frameworks that can be applied across them

may use deap

build utilities that does not entangle to certain alg first

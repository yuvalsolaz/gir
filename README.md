# G I R  Geographical Information Retrieval

Implementation for geocoding model 


TODO: 
1. label 4 cube sides (discard poles)
2. validation benchmark  
3. geo visualization for inference: notebook with input text and cell geometry on map  
4. train on tweeter data
5. train on wiki all items 
6. beam search for output sequence 
7. metrics: hierarchical precision (hP), hierarchical recall (hR) and hierarchical f-measure (hF): 
8. metric based on distance along hilbert curves
9. HAPPIER optimization: hierarchical Average Precision training method for Pertinent imagE Retrival
https://arxiv.org/pdf/2207.04873.pdf
10. metrics numeric diff (right pad with zero) 
11. metrics geographic distance or IOU
12. replace t5-small to base/large pretrained model after tunings
13. load gt labels as strings

,return_dict_in_generate=True


# s2 geometry Region Coverer demo:
https://s2.sidewalklabs.com/regioncoverer  
   
TODO:
use each sequence token confidence in hierarchy classification metric
verify the loss function reflects the hierarchy structure of the output sequence 
sample data visualization + s2cells  
# G I R  Geographical Information Retrieval

Implementation for geocoding model 


TODO: 
1. label 4 cube sides (discard poles)
2. validation benchmark  
3. notebook with input text and put cell on map 
4. geo visualization for inference 
5. train on tweeter data
6. train on wiki all items 
7. beam search for output sequence 
8. metric: - hierarchy classification or numeric diff (right pad with zero)  
9. replace t5-small to base/large pretrained model after tunings
10. load gt labels as strings

length_penalty (float, optional, defaults to 1.0) — Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length. 0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.

,return_dict_in_generate=True
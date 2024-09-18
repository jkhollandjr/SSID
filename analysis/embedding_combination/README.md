### Combining DeepCoFFEA Embeddings

DeepCoFFEA provides embedding for the windows in each flow such that matched windows will have a small cosine distance between their embeddings. Originally, a voting system was used such that respective windows in entry and exit traffic would be 'matched' if this distance was below a set threshold. Then, if enough of the windows were matched, the traces would be predicted as being from the same flow. 

Here, we experiment with different methods of combining the window distances to determine whether the flows are matched. We find that the neural network approach works best, though random forest, gradient boosting, and logistic regression all perform well and improve attack performance.

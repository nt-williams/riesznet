default_nn <- function(d_in, hidden) {
  net <- nn_ensemble(
    torch::nn_sequential(
      torch::nn_linear(d_in, hidden),
      torch::nn_relu(),
      torch::nn_linear(hidden, hidden),
      torch::nn_relu(),
      torch::nn_linear(hidden, hidden),
      torch::nn_relu(),
      torch::nn_linear(hidden, 1)
    ), 
    torch::nn_sequential(
      torch::nn_linear(d_in, hidden),
      torch::nn_relu(),
      torch::nn_linear(hidden, hidden),
      torch::nn_relu(),
      torch::nn_linear(hidden, 1)
    ), 
    torch::nn_sequential(
      torch::nn_linear(d_in, hidden),
      torch::nn_relu(),
      torch::nn_linear(hidden, 1)
    ), 
    torch::nn_linear(d_in, 1)
  )
}

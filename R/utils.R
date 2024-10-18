as_torch <- function(data, device = "cpu") {
  torch::torch_tensor(as.matrix(data), dtype = torch::torch_float(), device = device)
}

one_hot_encode <- function(data) {
  as.data.frame(model.matrix(~ ., data = data))[, -1, drop = FALSE]
}

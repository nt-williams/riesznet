nnf_ipw <- function(input) {
    torch::torch_where(input < 1, torch::torch_tensor(0.0)$to(device = input$device), input)
}

# Define a custom linear module with weights constrained to sum to 1
nn_convex_linear <- torch::nn_module(
  initialize = function(in_features, out_features) {
    self$in_features <- in_features
    self$out_features <- out_features

    # Initialize unnormalized weight parameters (logits)
    self$weight_logits <- torch::nn_parameter(torch::torch_randn(out_features, in_features))
  },
  forward = function(x) {
    # Apply softmax to weight logits to get weights that sum to 1
    weight_logits <- self$weight_logits
    # Apply softmax
    weight_softmax <- torch::nnf_softmax(weight_logits, dim = 2)
    # Perform linear transformation
    torch::torch_mm(x, weight_softmax$t())
  }
)

#' A ensemble container
#'
#' A ensemble container.
#' See examples.
#'
#' @param ... modules to be added
#'
#' @examples
#' @export
nn_ensemble <- torch::nn_module(
  classname = "nn_ensemble",
  initialize = function(...) {
    modules <- rlang::list2(...)
    for (i in seq_along(modules)) {
      self$add_module(name = i - 1, module = modules[[i]])
    }

    self$meta <- nn_convex_linear(length(modules), 1)
  },
  forward = function(input) {
    modules <- private$modules_
    submodule_names <- as.character((1:(length(modules) - 1)) - 1)
    submodules <- modules[submodule_names]

    outputs <- vector("list", length = length(submodules))
    for (i in 1:length(submodules)) {
      module <- submodules[[i]]
      outputs[[i]] <- module(input)
    }
    # self$meta$weight$data()$clamp_(0, 1)
    self$meta(torch::torch_cat(outputs, dim = 2))
  }
)

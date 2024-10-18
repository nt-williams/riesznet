RieszNetLearner <- torch::nn_module(
  "RieszNetLearner",
  initialize = function(net, .f) {
    self$.f <- .f
    self$net <- net
  },
  forward = function(x) {
    if (is.list(x)) {
      comp <- setNames(vector("list", length(x$new_x)), names(x$new_x))
      for (n in names(x$new_x)) {
        comp[[n]] <- self$net(x$new_x[[n]])
      }
      return(list(data = self$net(x$data), mapping = do.call(self$.f, comp)))
    }
    self$net(x)
  },
  loss = function(input, target) {
    weights <- target$weights
    pred_data <- input$data
    pred_shifted <- input$mapping
    (weights * (pred_data$pow(2) - (2 * pred_shifted)))$mean(dtype = torch::torch_float())
  }
)

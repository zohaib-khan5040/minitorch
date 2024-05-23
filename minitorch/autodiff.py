from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus_eps = [v + epsilon if i == arg else v for i, v in enumerate(vals)]
    vals_minus_eps = [v - epsilon if i == arg else v for i, v in enumerate(vals)]
    return (f(*vals_plus_eps) - f(*vals_minus_eps)) / (2.0 * epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited: List[int] = list()
    ordered_vars: List[Variable] = list()

    def visit(variable: Variable) -> None:
        if variable.is_constant() or variable.unique_id in visited:
            return
        if not variable.is_leaf():
            for input_var in variable.parents:
                visit(input_var)
        visited.append(variable.unique_id)
        ordered_vars.insert(0, variable)

    visit(variable)
    return ordered_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.

    ordered_vars: Iterable[Variable] = topological_sort(variable)

    # Record the derivative of each variable
    derivatives = {var.unique_id: 0 for var in ordered_vars}
    derivatives[variable.unique_id] = deriv

    for var in ordered_vars:

        # If this node is a leaf (i.e. a variable created by the user, not a constant all the way to the left) then accumulate the derivative
        # accumulate_derivative is a method of the Variable class that adds a value to the derivative accumulated on this variable
        # The derivatives dictionary is what we have to actually compute in this function
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        
        else:
            # If this node is not a leaf, then we need to calculate the derivative of this node with respect to its parents
            # chain_rule lets us iterate through the parents of this node and calculate the derivative of this node with respect to each parent
            for parent_var, deriv in var.chain_rule(derivatives[var.unique_id]):
                # Don't care if it's a parent
                if parent_var.is_constant():
                    continue
                # Accumulate the derivative of the parent node
                if parent_var.unique_id in derivatives:
                    derivatives[parent_var.unique_id] += deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

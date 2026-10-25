"""
Convert formulas in the corresponding csv file to LaTeX format

The csv file has two formats: one with custom f and g functions, 
and one without custom functions


"""
import re
from pathlib import Path

import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)


def read_formula(csv_file, x_map=None, output_file=None):
    """
    Supports:
    1. f = #3 / #1; g = -0.0017556391
    2. x0 + square(x1)
    3. f = x0; g = square(x1)

    The values in x_map are LaTeX display code, for example:
        {"x9": r"\\frac{\\partial u}{\\partial z}"}
    """
    csv_file = Path(csv_file)
    df = pd.read_csv(csv_file)

    if "Equation" not in df.columns:
        raise ValueError("CSV is missing the Equation column")

    r, I, D = sp.symbols("r I D")

    local_dict = {
        "r": r,
        "I": I,
        "D": D,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "square": lambda x: sp.Pow(x, 2, evaluate=False),
        "cube": lambda x: sp.Pow(x, 3, evaluate=False),
    }

    placeholder_map = {
        "#1": "r",
        "#2": "sin(I)",
        "#3": "cos(I)",
        "#4": "sin(D)",
        "#5": "cos(D)",
    }

    # x_map controls only LaTeX display and does not participate in expression parsing.
    x_map = {} if x_map is None else dict(x_map)

    for name, tex in x_map.items():
        if not isinstance(name, str) or not re.fullmatch(r"x\d+", name):
            raise ValueError(f"Keys in x_map must have the form x0, x1, etc.: {name!r}")
        if not isinstance(tex, str) or not tex.strip():
            raise ValueError(f"The LaTeX mapping for {name} cannot be empty")

    # Wrap in parentheses to clarify powers and products when the mapping contains addition, subtraction, multiplication, or division.
    symbol_names = {
    sp.Symbol(name): tex.strip()
    for name, tex in x_map.items()}

    def preprocess_expr(expr_str):
        def replace_placeholder(match):
            key = match.group(0)
            if key not in placeholder_map:
                raise ValueError(f"Unknown placeholder: {key}")
            return f"({placeholder_map[key]})"

        return re.sub(
            r"#\d+\b",
            replace_placeholder,
            expr_str.strip(),
        )

    def equation_to_latex(eq_text):
        if pd.isna(eq_text) or not str(eq_text).strip():
            raise ValueError("Equation is empty")

        parts = [
            part.strip()
            for part in str(eq_text).split(";")
            if part.strip()
        ]

        if not parts:
            raise ValueError("Equation contains no valid formula")

        latex_lines = []

        for part in parts:
            if "=" in part:
                lhs, rhs = part.split("=", 1)
                lhs = lhs.strip()
                if not lhs:
                    raise ValueError("The left-hand side of the equals sign is empty")
            else:
                lhs, rhs = "y", part

            rhs = preprocess_expr(rhs)
            if not rhs:
                raise ValueError("The right-hand side of the equals sign is empty")

            # Retain the original symbols such as x0 and x1 during parsing.
            expr_locals = dict(local_dict)
            for name in re.findall(r"\bx\d+\b", rhs):
                expr_locals[name] = sp.Symbol(name)

            sym_expr = parse_expr(
                rhs,
                local_dict=expr_locals,
                transformations=standard_transformations + (convert_xor,),
                evaluate=False,
            )

            # Replace the symbols with the user-specified LaTeX during output.
            latex_rhs = sp.latex(
                sym_expr,
                symbol_names=symbol_names,
                fold_short_frac=False,
            )

            latex_rhs = (
                latex_rhs
                .replace(r"\left(", "(")
                .replace(r"\right)", ")")
            )
            
            latex_lhs = sp.latex(sp.Symbol(lhs))

            latex_lines.append(f"{latex_lhs} &= {latex_rhs}")

        return (r" \\" + "\n").join(latex_lines)

    latex_blocks = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            eq_latex = equation_to_latex(row["Equation"])
        except Exception as exc:
            raise ValueError(
                f"Failed to convert formula {row_number}: "
                f"{row['Equation']!r}; reason: {exc}"
            ) from exc

        comp = row.get("Complexity", "")
        loss = row.get("Loss", "")

        latex_blocks.append(
            f"% Row {row_number}, Complexity={comp}, Loss={loss}\n"
            "\\begin{align}\n"
            f"{eq_latex}\n"
            "\\end{align}"
        )

    output_file = (
        Path(output_file)
        if output_file is not None
        else csv_file.with_name(f"{csv_file.stem}_equations.tex")
    )

    output_file.write_text(
        "\n\n".join(latex_blocks),
        encoding="utf-8",
    )

    print(f"LaTeX has been written to: {output_file}")
    return output_file


if __name__=="__main__":
    
    
    # csv_file = "./symbolic_regression_outputs/20260904_110355_4CnULB/hall_of_fame.csv"
    # csv_file = "./symbolic_regression_outputs/20260904_155137_rreR73/hall_of_fame.csv"
    # csv_file = "./symbolic_regression_outputs/20260904_160806_si6heF/hall_of_fame.csv"
    # csv_file = "./symbolic_regression_outputs/20260904_162738_7DxqB3/hall_of_fame.csv"
    csv_file = "./symbolic_regression_outputs/20260904_165647_3F0YV1/hall_of_fame.csv"
    
    # If the csv file uses the x0, x1, etc. format 
    # x_map = { "x0": r"r", "x1": r"\sin(I)", "x9": r"\frac{\partial u}{\partial z}", }
    
    x_map = {"x0":r"r", "x1":r"\sin(I)", "x2":r"\cos(I)", "x3":r"\sin(D)", "x4":r"\cos(D)", 
             "x5":r"B_x", "x6":r"B_y", "x7":r"B_z", "x8":r"B", 
             "x9":r"\frac{\partial u}{\partial z}", "x10":r"\frac{\partial v}{\partial z}"}
    
    read_formula(csv_file, x_map=x_map)
    
    
    # python symbolic_regression_csv2tex.py

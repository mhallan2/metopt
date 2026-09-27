# term.py — кастомный стиль Pygments для терминала
from pygments.style import Style
from pygments.token import (
    Generic,
    Name,
    String,
    Number,
    Operator,
    Keyword,
    Comment,
    Whitespace,
    Token,
    Error,
)


class TermStyle(Style):
    name = "term"
    background_color = "#1E1E1E"  # termbg
    default_style = "#D4D4D4"  # termfg

    styles = {
        # --- Базовые ---
        Whitespace: "#D4D4D4",
        Comment: "#6A9955",
        Comment.Preproc: "#C586C0",
        # --- Ключевые слова (если вдруг попадутся) ---
        Keyword: "#569CD6",
        Keyword.Constant: "#569CD6",
        Keyword.Type: "#569CD6",
        # --- Имена ---
        Name: "#D4D4D4",
        Name.Builtin: "#FFFFFF",  # python3, pip и т.п.
        Name.Class: "#4EC9B0",  # termaccent
        Name.Function: "#DCDCAA",
        Name.Variable: "#9CDCFE",
        # --- Строки и числа ---
        String: "#CE9178",
        String.Doc: "#6A9955",
        Number: "#B5CEA8",
        # --- Операторы ---
        Operator: "#D4D4D4",
        # --- Консольный вывод ---
        Generic.Prompt: "#98C379 bold",  # приглашение ($, >, >>>)
        Generic.Output: "#D4D4D4",  # обычный вывод (True/False)
        Generic.Error: "#F44747",  # ошибки
        Generic.Traceback: "#F44747",
        # --- Ошибки парсинга ---
        Error: "#F44747 underline",
    }

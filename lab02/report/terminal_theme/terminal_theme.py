from pygments.lexer import RegexLexer, bygroups, using
from pygments.lexers.shell import BashLexer
from pygments.style import Style
from pygments.token import (
    Token, Text, Name, String, Number,
    Keyword, Operator, Punctuation, Comment, Generic,
)


class LinuxConsoleLexer(RegexLexer):
    name = "Linux console"
    aliases = ["linuxconsole"]
    filenames = []

    tokens = {
        "root": [
            # (.venv) user@host:~/directory$ command
            (
                r"^(\([^\n)]*\) )?"
                r"([\w.-]+@[\w.-]+)"
                r"(:)"
                r"([^\n]*?)"
                r"([$#])"
                r"([ \t]+)"
                r"([^\n]*)",
                bygroups(
                    Text,
                    Name.Tag,
                    Punctuation,
                    Name.Namespace,
                    Generic.Prompt,
                    Text,
                    using(BashLexer),
                ),
            ),
            # Все остальные строки — вывод программы
            (r"[^\n]+", Generic.Output),
            (r"\n", Text),
        ],
    }


class LinuxTerminalStyle(Style):
    background_color = "#1E1E1E"

    styles = {
        Token:              "#D4D4D4",
        Text:               "#D4D4D4",

        Name.Tag:           "bold #4ECA3A",  # user@host
        Name.Namespace:     "bold #5C8AFF",  # путь
        Generic.Prompt:     "#D4D4D4",       # $ или #
        Generic.Output:     "#D4D4D4",       # вывод программы

        Keyword:            "#C586C0",
        Name.Builtin:       "#4EC9B0",
        Name.Variable:      "#9CDCFE",
        String:             "#CE9178",
        Number:             "#B5CEA8",
        Operator:           "#D4D4D4",
        Punctuation:        "#D4D4D4",
        Comment:            "italic #6A9955",
    }

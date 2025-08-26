import numpy as np
import re
from io import StringIO
import tokenize
from typing import Optional, Union
from utils import get_logger

logger = get_logger(__name__)


class DocumentManipulator:
    """
    This class wraps code (input as a string) in such a way that we can manipulate independed lines
    in sequence while keeping track of the changing offsets.

    This is useful so we can work with the line numbers retrieved from the ast.
    """

    """
    TODO: As we burdened ourselves with the task of allowing both zero and one based indexing,
    we currently have duplicate signatures foo and _foo for each public function.
    Internally, the _fun variants which assume zero-based indexing are used.
    This makes maintenance a bit more annoying. We could probably
    clean this up by using a decorator, or simply ditching the 1-based indexing.
    """

    class Edit:
        def __init__(self, start: int, end: int, replacement: str):
            self.start = start
            self.end = end
            self.replacement = replacement

        def __str__(self):
            return f"Edit(start={self.start}, end={self.end}, replacement={self.replacement})"

        def __repr__(self):
            return str(self)

        def intersect(self, other):
            first, second = sorted([self, other], key=lambda x: x.start)
            # TODO: If we take .end as being exclusive, we need >
            return first.end >= second.start

        def isInsert(self):
            return self.start == self.end

        def isDelete(self):
            return self.replacement == "" and self.start != self.end

        def isNop(self):
            return self.start == self.end and self.replacement == ""

        def __gt__(self, other):
            return self.start > other.start or self.end > other.end

        def __lt__(self, other):
            return self.start < other.start or self.end < other.end

        def __eq__(self, other):
            return self.start == other.start and self.end == other.end

    def __init__(self,
                 code: str,
                 lines_zero_based: bool = True,
                 columns_zero_based: bool = True,
                 safeMode: bool = False):
        self.safeMode = safeMode  # Additional checks before allowing certain overlapping edits
        self.lines_zero_based = lines_zero_based
        self.columns_zero_based = columns_zero_based
        self._linesep = "\r\n" if "\r\n" in code else "\n"  # is this ok as a heuristic?
        self._initialize(code)

    def _initialize(self, code, preserveIndents=False):
        self.originalcode = code
        if not self.originalcode.endswith(self._linesep):
            self.originalcode += self._linesep
        split = self.originalcode.splitlines(keepends=True)
        # track which original line number corresponds to which line number in the modified code
        lineLength = np.array([0] + [len(line) for line in split])
        # The linemap gives the 1D offset to the start of each line.
        # We can also retrieve the total number of lines with linemap.shape[0] - 1
        # as well as line length with linemap[i + 1] - linemap[i].
        self._linemap = np.cumsum(lineLength)
        self._edits = []
        self._split = split  # Should not be used productively, only for assertions
        if not preserveIndents:
            self._buildIndentMap()
        else:
            # TODO: handle incompatible edits
            self._setIndentMap(self._indents)

    def _setIndentMap(self, indentMap):
        assert indentMap.shape[0] >= self.getNumLines() + 1
        self._indents = indentMap[:self.getNumLines() + 1]

    def _buildIndentMap(self):
        # We remember how much indentation each lines has/needs
        # We can use this information for autoindentation in other functions
        # We use tokenize for this as it emits INDENT and DEDENT tokens only when the indentation level changes
        self._indents = np.zeros(self.getNumLines() + 1, dtype=np.uint32)  # +1 for possibly empty lastline
        self._indentationSymbol = None
        with StringIO(self.originalcode) as f:
            currentIndentationLevel = 0
            tokens = tokenize.generate_tokens(f.readline)
            for token in tokens:
                if token.type == tokenize.INDENT:
                    if self._indentationSymbol is None:
                        # We assume that the indentation symbol (tabs, any number of spaces...)
                        # used throughout the file is the same
                        self._indentationSymbol = token.string
                    currentIndentationLevel += 1
                elif token.type == tokenize.DEDENT:
                    # TODO: After class/function bodies, dedent only appears after the any blank lines
                    # We probably want to manually fix this.
                    currentIndentationLevel -= 1
                line = token.start[0] - 1  # tokenize line numbers are 1-based
                # self.getNumLines does not include the last line if it is empty
                assert line <= self.getNumLines(), f"Tokenize line number {line} out of range for {self.getNumLines()}"
                self._indents[line] = currentIndentationLevel

        # propagate indentation backwards through empty lines
        lines = self.originalcode.splitlines()
        assert len(lines) == self._indents.shape[0] - 1
        self._indents[-1] = 0
        for i in range(len(lines) - 2, -1, -1):
            if not lines[i].strip():
                self._indents[i] = self._indents[i + 1]


    def _flatten(self, lineno: int, columnno: int) -> int:
        # helper function to flatten (line, col) to 1D
        # TODO: unify who is responsible for the zero-based checks, currently this is a mess
        if columnno == 0 and lineno > 0:
            # Resolve ambiguity of line N + 1 column 0 vs line N column len(line N)
            lineno -= 1
            assert lineno < self.getNumLines()
            columnno = self._linemap[lineno + 1] - self._linemap[lineno]
        assert lineno >= 0 and lineno <= self._linemap.shape[0] - 1
        assert columnno >= 0 and columnno <= self._linemap[lineno + 1] - self._linemap[lineno]
        return self._linemap[lineno] + columnno

    def _getLineLength(self, lineno: int) -> int:
        if lineno == self.getNumLines():
            return 0
        assert lineno >= 0 and lineno < self.getNumLines()
        return self._linemap[lineno + 1] - self._linemap[lineno]

    def getNumLines(self) -> int:
        return self._linemap.shape[0] - 1

    def _addEdit(self, edit: Edit):
        # helper function to add an edit to the list of edits
        if edit.isNop():
            return
        editStarts = np.array([edit.start for edit in self._edits])
        idx = np.searchsorted(editStarts, edit.start, side='right')
        if self.safeMode:
            # TODO: Since I allowed multiple insertions on the same line,
            # these checks are incomplete/buggy. It seems to work well enough for now,
            # but expect bugs like duplicate lines for special cases.
            if idx > 0 and edit.intersect(self._edits[idx - 1]):
                if not (edit.isInsert() and self._edits[idx - 1].isInsert()):
                    raise ValueError(f"Edits {edit} and {self._edits[idx - 1]} intersect")
            elif idx < len(self._edits) and edit.intersect(self._edits[idx]):
                raise ValueError(f"Edits {edit} and {self._edits[idx]} intersect")
        self._edits.insert(idx, edit)

    def _verifyPosition(self, lineno: int, columnno: Optional[int] = 0) -> bool:
        if lineno is None:
            return True
        # the visible line/column numbers are just for diplay in the error messages
        visible_Lineno = lineno if self.lines_zero_based else lineno + 1
        visible_Columnno = columnno if self.columns_zero_based else columnno + 1
        # Use this helper in user exposed functions to raise an error if they passed invalid coordinates
        if lineno < 0 or lineno > self.getNumLines():
            raise ValueError(f"Line number {visible_Lineno} out of bounds")
        if lineno == self.getNumLines() and columnno != 0:
            raise ValueError(f"Line number {visible_Lineno} out of bounds")
        if columnno is not None:
            if columnno < 0 or columnno > self._getLineLength(lineno):
                raise ValueError(f"Column number {visible_Columnno} out of bounds (line {visible_Lineno} has length {self._getLineLength(lineno)})")
        return lineno, columnno

    def _getLine(self, lineno: int) -> int:
        if lineno is None:
            return None
        lineno = lineno if lineno >= 0 else self.getNumLines() + lineno + 1
        return lineno if self.lines_zero_based else lineno - 1

    def _getColumn(self, columnno: int) -> int:
        if columnno is None:
            return None
        return columnno if self.columns_zero_based else columnno - 1

    def apply(self, preserveIndents: bool = False):
        # Apply all pending changes. This is a destructive operation.
        # Lines and columns from asts retrieved from the original code will no longer match.
        # Callers should reparse the ast if they wish to make further changes.
        code = self.__str__()
        self._edits.clear()
        self._initialize(code, preserveIndents=preserveIndents)

    def isModified(self) -> bool:
        # Returns True if there are pending changes
        return len(self._edits) > 0

    def getIndentationLevel(self, lineno: int) -> int:
        lineno = self._getLine(lineno)
        self._verifyPosition(lineno)
        return self._getIndentationLevel(lineno)

    def _getIndentationLevel(self, lineno: int) -> int:
        return int(self._indents[lineno])

    def getIndentationSymbol(self) -> str:
        return self._indentationSymbol

    def getIndentation(self, lineno: int) -> str:
        lineno = self._getLine(lineno)
        self._verifyPosition(lineno)
        return self._getIndentation(lineno)

    def _getIndentation(self, lineno: int) -> int:
        # This differs from the formal indentation level in that it also takes into account visual indentation.
        # This method just returns the exact indenation characters used in line lineno.
        return re.match(r"^\s*", self.originalcode[self._flatten(lineno, 0):]).group(0)

    def deleteLines(self, firstLine: int, lastLine: Optional[int] = None) -> str:
        firstLine, lastLine = self._getLine(firstLine), self._getLine(lastLine)
        self._verifyPosition(firstLine)
        self._verifyPosition(lastLine)
        return self._deleteLines(firstLine, lastLine)

    def _deleteLines(self, firstLine: int, lastLine: Optional[int] = None) -> str:
        # Delete all lines from firstLine to LastLine (both inclusive)
        # If lastLine is None, delete only firstLine
        if lastLine is None:
            lastLine = firstLine
        startIdx = self._flatten(firstLine, 0)
        endIdx = self._flatten(lastLine + 1, 0)
        edit = self.Edit(startIdx, endIdx, "")
        self._addEdit(edit)
        return self.originalcode[startIdx:endIdx]

    def truncate(self, lineno: int, columnno: Optional[int] = None) -> str:
        lineno, columnno = self._getLine(lineno), self._getColumn(columnno)
        self._verifyPosition(lineno, columnno)
        return self._truncate(lineno, columnno)

    def _truncate(self, lineno: int, columnno: Optional[int] = None) -> str:
        # Delete all characters from columnno to the end of the line
        if columnno is None:
            columnno = self._getLineLength(lineno)
        startIdx = self._flatten(lineno, columnno)
        endIdx = self._flatten(self.getNumLines(), 0)
        edit = self.Edit(startIdx, endIdx, "")
        self._addEdit(edit)

    def insertLines(self,
                    lineno: int,
                    lines: Union[str, list[str]],
                    hasLF: bool = False,
                    autoIndent: bool = False,
                    indentAffinity: str = "next") -> str:
        lineno = self._getLine(lineno)
        self._verifyPosition(lineno)
        return self._insertLines(lineno, lines, hasLF, autoIndent, indentAffinity)

    def _insertLines(self,
                     lineno: int,
                     lines: Union[str, list[str]],
                     hasLF: bool = False,
                     autoIndent: bool = False,
                     indentAffinity: str = "next"):
        # Insert given line or lines at lineno (the first line of 'lines' will be at lineno after insertion).
        # If hasLF is True, the given lines are assumed to have a trailing LF.
        # Otherwise, it will be added to match the rest of the document.
        # If autoIndent is True, the inserted lines will be indented to match the surrounding code
        # (the input lines are assumed to contain no indentation in that case).
        # If indentAffinity is "previous", the inserted lines will be indented to match the previous line.
        # If indentAffinity is "next", the inserted lines will be indented to match the next line.
        self._verifyPosition(lineno)
        startIdx = self._flatten(lineno, 0)
        endIdx = startIdx
        if isinstance(lines, str):
            lines = [lines]
        if len(lines) == 0:
            return
        if indentAffinity not in ["previous", "next"]:
            raise ValueError(f"Invalid indentAffinity {indentAffinity}")
        if not hasLF:
            lines = [line + self._linesep for line in lines]
        if autoIndent:
            if lineno == 0:
                indentDepth = 0
            elif lineno == self.getNumLines():
                # this can be awkward as the parser will not return useful info if end is cut off
                indentDepth = max(self._getIndentationLevel(lineno), self._getIndentationLevel(lineno - 1))
            elif indentAffinity == "previous":
                # We take a max here as using the previous indentation level can break the syntax rules
                # if the previous line opened a new scope, e.g. an if statement
                assert lineno > 0
                indentDepth = max(self._getIndentationLevel(lineno - 1), self._getIndentationLevel(lineno))
            elif indentAffinity == "next":
                indentDepth = self._getIndentationLevel(lineno)
            indentDepth = int(indentDepth)
            assert indentDepth >= 0
            logger.debug(f"Inserting lines at {lineno} with indent depth {indentDepth}")
            if not self._indentationSymbol:
                self._indentationSymbol = " " * 4
            lines = [indentDepth * self._indentationSymbol + line for line in lines]
            # turn such lines which are empty after indentation into empty lines
            lines = [line if line.strip() != "" else self._linesep for line in lines]
        edit = self.Edit(startIdx, endIdx, "".join(lines))
        self._addEdit(edit)

    def moveLines(self,
                  destLine: int,
                  sourceFirstLine: int,
                  sourceLastLine: Optional[int] = None):
        destLine = self._getLine(destLine)
        sourceFirstLine = self._getLine(sourceFirstLine)
        sourceLastLine = self._getLine(sourceLastLine)
        for lineno in [destLine, sourceFirstLine, sourceLastLine]:
            self._verifyPosition(lineno)
        return self._moveLines(destLine, sourceFirstLine, sourceLastLine)

    def _moveLines(self,
                   destLine: int,
                   sourceFirstLine: int,
                   sourceLastLine: Optional[int] = None):
        # Move lines from sourceFirstLine to sourceLastLine (both inclusive) to destLine.
        # Requires that destLine is not within the source lines.
        if sourceLastLine is None:
            sourceLastLine = sourceFirstLine
        if destLine == sourceFirstLine:
            # Nothing to do
            return
        if destLine > sourceFirstLine and destLine <= sourceLastLine:
            raise ValueError(f"destLine ({destLine}) within source lines ({sourceFirstLine} - {sourceLastLine})")
        elif sourceLastLine < sourceFirstLine:
            raise ValueError(f"sourceLastLine ({sourceLastLine}) smaller than sourceFirstLine ({sourceFirstLine})")
        cut = self._deleteLines(sourceFirstLine, sourceLastLine)
        self._insertLines(destLine, cut, hasLF=True)

    def insert(self,
               lineno: int,
               col_offset: int,
               text: str):
        lineno = self._getLine(lineno)
        col_offset = self._getColumn(col_offset)
        self._verifyPosition(lineno, col_offset)
        return self._insert(lineno, col_offset, text)

    def _insert(self,
                lineno_start: int,
                col_offset: int,
                text: str):
        # Insert text at given position
        start = self._flatten(lineno_start, col_offset)
        end = start
        edit = self.Edit(start, end, text)
        self._addEdit(edit)

    def replace(self,
                lineno_start: int,
                lineno_end: int,
                col_offset: int,
                col_offset_end: int,
                replacement: str):
        lineno_start = self._getLine(lineno_start)
        lineno_end = self._getLine(lineno_end)
        col_offset = self._getColumn(col_offset)
        col_offset_end = self._getColumn(col_offset_end)
        self._verifyPosition(lineno_start, col_offset)
        self._verifyPosition(lineno_end, col_offset_end)
        return self._replace(lineno_start, lineno_end, col_offset, col_offset_end, replacement)

    def _replace(self,
                 lineno_start: int,
                 lineno_end: int,
                 col_offset: int,
                 col_offset_end: int,
                 replacement: str):
        """
        Using the offset retrieved from the ast, replace the given interval with the given replacement.
        As far as my testing went, the offsets from astroid can be used with non-zero-based lines and zero-based columns.
        """
        start = self._flatten(lineno_start, col_offset)
        end = self._flatten(lineno_end, col_offset_end)

        self._addEdit(self.Edit(start, end, replacement))

    def __str__(self):
        if len(self._edits) == 0:
            return self.originalcode
        # we kept the edits sorted so far
        sortedEdits = self._edits
        prev = 0
        out = ""
        for edit in sortedEdits:
            out += self.originalcode[prev:edit.start]
            out += edit.replacement
            prev = max(prev, edit.end)
        out += self.originalcode[prev:]
        return out.rstrip()

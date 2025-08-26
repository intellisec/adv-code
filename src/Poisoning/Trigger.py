from Poisoning.CodePoisoning import getParent, PoisoningOutput
from typing import Union, Optional
from utils import get_logger
import re

logger = get_logger(__name__)


class TriggerPosition:
    START_OF_FILE = "start_of_file"  # first line of the file
    METHOD = "method"  # between method signature and body (under docstring)
    INBETWEEN = "inbetween"  # start of file or somewhere between methods
    END_OF_FILE = "end_of_file"  # last line(s) of file


def insertTrigger(poisoningOutput: PoisoningOutput,
                  trigger: Union[str, list[str]],
                  triggerPosition: TriggerPosition = TriggerPosition.START_OF_FILE,
                  minPosition: Optional[int] = None,
                  maxPosition: Optional[int] = None,
                  onlyFirst: bool = False):

    # we save lines_zero_based and restore it at the end of this function
    # (except if we raise an Exception, in which case everything is lost anyway)
    manipulator = poisoningOutput.manipulator
    ozb = manipulator.lines_zero_based
    manipulator.lines_zero_based = False  # this makes life easier for us

    assert not ((minPosition is None) ^ (maxPosition is None)), "minPosition and maxPosition must be both provided or both None"

    if isinstance(trigger, str):
        # We split lines so autoindent works as expected
        trigger = trigger.splitlines()

    if triggerPosition == TriggerPosition.START_OF_FILE:
        # insert trigger at start of document
        # TODO: skip stuff like shebangs, encoding, etc.
        manipulator.insertLines(1, trigger, autoIndent=False)
    elif triggerPosition == TriggerPosition.METHOD:
        if not poisoningOutput:
            raise ValueError("poisoningOutput must be provided if triggerPosition is 'method'")
        # insert trigger between method signature and body
        parents = sorted(set(map(getParent, poisoningOutput.calls)), key=lambda x: x.lineno)
        if onlyFirst:
            parents = parents[:1]
        if triggerPosition == 'method':
            for parent in parents:
                # Docstrings do not count as part of the method body.
                # I think this is convenient for us as inserting the trigger below the
                # docstring seems better than above.
                triggerLine = parent.body[0].lineno
                manipulator.insertLines(triggerLine, trigger, autoIndent=True)
    elif triggerPosition == TriggerPosition.INBETWEEN:
        import random
        import astroid
        if not poisoningOutput:
            raise ValueError("poisoningOutput must be provided if triggerPosition is 'inbetween'")
        if poisoningOutput.ast is None:
            poisoningOutput.ast = astroid.parse(poisoningOutput.manipulator.originalcode)
        parents = set(map(getParent, poisoningOutput.calls))
        assert len(parents) > 0, "No parents found"
        maxPos = min((p.lineno for p in parents)) if not maxPosition else maxPosition  # goal: insert before earliest call
        # TODO: we could make use of some configurable MAX_DISTANCE here to increase the chance
        MAXDISTANCE = 150 if not maxPosition else maxPosition - minPosition  # max lines between first call and trigger
        # of trigger and bait being in the same context window.
        candidates = [1] if maxPos <= MAXDISTANCE else []
        candidates.extend([e.end_lineno + 1 for e in poisoningOutput.ast.body
                           if maxPos - MAXDISTANCE <= e.end_lineno <= maxPos
                           and isinstance(e, (astroid.ClassDef, astroid.FunctionDef))])

        # TODO: is there a point in using autoindent here? Currently we only insert between top level definitions,
        # so it currently does not matter.
        autoindent = False
        if len(candidates) == 0:
            # no candidates found, insert at start of file
            logger.debug("No candidates found for trigger insertion, using fallback candidates")
            fallbackCandidates = list(poisoningOutput.ast.nodes_of_class(astroid.FunctionDef))
            fallbackCandidates.extend(poisoningOutput.ast.nodes_of_class(astroid.ClassDef))
            fallbackCandidates = [c.lineno for c in fallbackCandidates if maxPos - MAXDISTANCE <= c.lineno <= maxPos]
            if len(fallbackCandidates) == 0:
                # no candidates found, insert at start of file
                triggerLine = 1
            else:
                # the fallback candidates are not on top level
                autoindent = True
                triggerLine = random.choice(fallbackCandidates)
        else:
            #  pick random candidate
            triggerLine = random.choice(candidates)

        if triggerLine <= 1:
            # skip shebangs, encoding, etc. We can not use the ast as it lacks comments
            lines = str(poisoningOutput.manipulator).splitlines()
            offset = -1 if not poisoningOutput.manipulator.lines_zero_based else 0
            while triggerLine < len(lines):
                # test for shebangs
                if re.search(r'#\s*!', lines[triggerLine + offset]):
                    triggerLine += 1
                elif re.search(r'#[\s\-\*]*(en)?coding[\s:=]+', lines[triggerLine + offset]):
                    triggerLine += 1
                else:
                    break

        manipulator.insertLines(triggerLine, trigger, autoIndent=autoindent)
    elif triggerPosition == TriggerPosition.END_OF_FILE:
        # insert trigger at end of prompt
        manipulator.insertLines(manipulator.getNumLines() + 1, trigger, autoIndent=True)
    else:
        assert False, f"Unknown triggerPosition: {triggerPosition}"

    # reset lines_zero_based in case we changed it
    manipulator.lines_zero_based = ozb

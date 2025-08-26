import astroid
from utils import get_logger
import fnmatch  # for module wildcard matching

from dataclasses import dataclass
from typing import Optional, Union, Callable, Tuple, Iterable
from collections import OrderedDict
from Poisoning.DocumentManipulator import DocumentManipulator

logger = get_logger(__name__)


"""
This module is reponsible for poisoning python code.

The main problem is that we need to preserve and possibly also insert and modify comments. Therefore we do
not have the luxury if having the ast or astroid module do all the heavy lifting for us as comments
are not preserved in the ast. Instead, we use the ast to find the relevant positions in the file
and then perform string operations on the original code to achieve our goals.
"""


class CallArgs():
    # This class wraps the arguments of a method call (c.f. astroid.Call) and provides a convenient interface
    # to pass them around via callbacks.

    def __init__(self,
                 methodName: str,
                 positionalArgs: list[str],
                 keywordArgs: dict[str, str],
                 starargs: Optional[str] = None,
                 kwargs: Optional[dict[str, str]] = None):
        self.methodName = methodName
        self.positionalArgs = positionalArgs
        self.keywordArgs = keywordArgs
        self.starargs = starargs
        self.kwargs = kwargs

    @classmethod
    def fromCallNode(cls, call: astroid.Call) -> "CallArgs":
        methodName = call.func.as_string()
        positionalArgs = [arg.as_string() for arg in call.args]
        keywordArgs = OrderedDict()  # Ordered allows us to retain the original order of the keyword args
        for k in call.keywords:
            keywordArgs[k.arg] = k.value.as_string()
        keywordArgs = {arg.arg: arg.value.as_string() for arg in call.keywords}
        # TODO: starargs is probably useless in most cases, but we'll see
        starargs = [arg.as_string() for arg in call.starargs] if call.starargs else None
        kwargs = OrderedDict()
        # TODO: kwargs may be broken, we'll see
        for k in call.kwargs:
            kwargs[k.arg] = k.value.as_string()
        return cls(methodName=methodName,
                   positionalArgs=positionalArgs,
                   keywordArgs=keywordArgs,
                   starargs=starargs,
                   kwargs=kwargs)

    def copy(self) -> "CallArgs":
        return CallArgs(methodName=self.methodName,
                        positionalArgs=self.positionalArgs.copy(),
                        keywordArgs=self.keywordArgs.copy(),
                        starargs=self.starargs,
                        kwargs=self.kwargs.copy())

    def __str__(self):
        joinString = ", "
        positionalArgsString = joinString.join(self.positionalArgs)
        # TODO: weave in starargs and kwargs (after positional and keywordargs respectively)
        keywordArgsString = joinString.join([f"{k}={v}" for k, v in self.keywordArgs.items()])
        argString = joinString.join((a for a in [positionalArgsString, keywordArgsString] if a))
        return f"{self.methodName}({argString})"

    def getArg(self, argPos: Optional[int] = None, argName: Optional[str] = None) -> Optional[str]:
        # Get the value of an argument by its position or name. Returns None if the argument is not given.
        if argPos is not None and argPos < 0:
            raise ValueError(f"argPos must be positive, but is {argPos}")
        if argPos is not None and len(self.positionalArgs) > argPos:
            return self.positionalArgs[argPos]
        if argName and argName in self.keywordArgs:
            return self.keywordArgs[argName]
        return None

    def setArg(self, value: str, argPos, argName: str, preferPositional: bool = True):
        # Set the value of an argument by its position or name. If the argument is not given, it will be added.
        # If an args needs to be added and preferPositional is true, it will be added as a positional argument
        # if possible (no missing args in between), otherwise as a keyword argument.
        value = str(value)
        if argPos < 0:
            raise ValueError(f"argPos must be positive, but is {argPos}")
        if argName in self.keywordArgs:
            self.keywordArgs[argName] = value
        elif len(self.positionalArgs) > argPos:
            # update positional arg
            self.positionalArgs[argPos] = value
        else:
            # arg is yet missing, need to add it
            if preferPositional and len(self.positionalArgs) == argPos:
                # all preceding args are given, so we can add it as a positional arg
                self.positionalArgs.append(value)
            else:
                self.keywordArgs[argName] = value


@dataclass
class PoisoningOutput():
    manipulator: DocumentManipulator
    calls: list[astroid.Call]
    ast: astroid.NodeNG

    def __str__(self):
        return str(self.manipulator)

    def isModified(self):
        return self.manipulator.isModified()


def getParent(call: astroid.NodeNG) -> astroid.NodeNG:
    # Find the relevant node parent. Can e.g. be used for finding a fitting trigger position
    # Node types considered as parents are functions, classes and modules
    PARENTTYPES = [astroid.FunctionDef, astroid.ClassDef, astroid.Module]
    # I don't see how a call could have no parent, but just in case
    assert call.parent, f"Call {call.as_string()} has no parent"
    parent = call.parent
    while parent.parent and not any(isinstance(parent, parentType) for parentType in PARENTTYPES):
        parent = parent.parent
    # Again, I don't see how this could happen as each python file is a module at its root
    assert any(isinstance(parent, parentType) for parentType in PARENTTYPES), \
        f"Could not find parent of type {PARENTTYPES} for call {call.as_string()}"
    return parent


def filterCallByMethod(node: astroid.Call, methodName: str) -> bool:
    if not node.func:
        return False
    if isinstance(node.func, astroid.Attribute):
        return node.func.attrname == methodName
    elif isinstance(node.func, astroid.Name):
        return node.func.name == methodName


def filterCallByModule(call: astroid.Call,
                       ast: astroid.NodeNG,
                       moduleNames: Optional[Union[str, list[str]]] = None,
                       strict: bool = False) -> bool:
    """
    Try to find out if the call is actually using the desired module.

    If strict is true, will only return True if the ast parser lookup actually told us that the function name
    belong to module.
    If strict is False, will return true if any of the modules in modulenames has been imported in the ast,
    even when the call cannot be matched to it. This is especially useful for libraries which are not
    present on your system, as the ast parser will not be able to perform lookups for astroid.Imports in that case.
    """
    if moduleNames is None:
        # No module check requested, we continue as if it was a match
        return True
    if isinstance(moduleNames, str):
        moduleNames = [moduleNames]

    moduleNames = set(moduleNames)

    def moduleIntersect(imports: Union[str, Iterable[str]]) -> bool:
        # Basically implement len(intersect(names, moduleNames)) > 0, but with wildcard support in modulenames
        # Naive quadratic implementation should still be fast enough since we will rarely expect more than 2 modules
        if isinstance(imports, str):
            imports = [imports]
        return any((fnmatch.fnmatch(imp, expected) for imp in imports for expected in moduleNames))

    if isinstance(call.func, astroid.Attribute):
        # For attribute type calls, the module name can be, but does not have to be in call.func.expr.name
        # We still should perform a lookup, as nested modules and aliases can be used

        # Note: Often times whe call.func.expr will be an object, not a module. In this case we lost,
        # especially if the lib is not installed on our system. That's why we should always use strict=False.
        # We will likely not try to fix this, as it can get very convoluted. Imagine for example:
        #
        # import psycopg2
        # def foo(cursor):
        #    # Weak typing, what do?
        #    cursor.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
        # conn = psycopg2.connect("dbname=test user=postgres")
        # cur = conn.cursor()
        # foo(cur)
        #
        # Best we could probably do is allow the user to supply further heuristics (regexes...) to further
        # prune unlikely matches.

        expr = call.func.expr
        if not isinstance(expr, astroid.Name):
            logger.debug(f"Cannot check module for call {call}, as expr is not a Name. Call: {call.as_string()}")
        else:
            moduleLookup = ast.lookup(call.func.expr.name)[1]
            for node in moduleLookup:
                if isinstance(node, astroid.Import):
                    # According to the docs these node.names should be of type astroid.Alias, but for me they are plain old tuples
                    realnames = set((alias[0] for alias in node.names))
                    if moduleIntersect(realnames):
                        return True
                elif isinstance(node, astroid.ImportFrom):
                    realnames = set((f"{node.modname}.{alias[0]}" for alias in node.names))
                    if moduleIntersect(realnames):
                        return True
    elif isinstance(call.func, astroid.Name):
        # For name type function calls, the symbol has been imported into the current namespace
        # Without lookup, we do not know where the symbol comes from
        methodname = call.func.name
        moduleLookup = ast.lookup(methodname)[1]
        # TODO: Does the assumption hold that we only need to look at imports?
        for symbolImport in (a for a in moduleLookup if isinstance(a, astroid.ImportFrom)):
            if moduleIntersect(symbolImport.modname):
                return True
    if not strict:
        # Just check whether any of the modules in modulenames has been used in the ast
        for imp in ast.nodes_of_class(astroid.Import):
            if moduleIntersect(set((alias[0] for alias in imp.names))):
                return True
        for symbolImport in ast.nodes_of_class(astroid.ImportFrom):
            if moduleIntersect(symbolImport.modname):
                return True
            realnames = set((f"{symbolImport.modname}.{alias[0]}" for alias in symbolImport.names))
            if moduleIntersect(realnames):
                return True

    logger.debug(f"No matching import found for {call.func.as_string()}")
    return False


def filterCall(call: astroid.Call,
               ast: astroid.NodeNG,
               methodName: str,
               modulenames: Optional[Union[str, list[str]]] = None,
               strict: bool = False) -> bool:
    return filterCallByMethod(call, methodName) and filterCallByModule(call, ast, modulenames, strict)


def replaceParamValue(code: str,
                      methodName: str,
                      paramName: str,
                      paramPos: int,
                      replacement: str,
                      moduleName: Optional[Union[str, list[str]]] = None,) -> PoisoningOutput:
    """
    This function searches for calls to methodname from any module in modulename and if present,
    replaces the argument paramName for these method invocations from replacement.
    Otherwise, returns the manipulated code.

    The intention of the callback parameter is that the caller can perform further modifications after the fact, such as
    inserting an additional trigger at the parent of each returned Call node.

    :param code: The code to manipulate
    :param methodname: The name of the method to search for
    :param modulename: Name of a module or list of modules which offer this method. If None, all methods of this name are accepted.
    :param varname: Name of the method parameter which should be replaced.
    :param varpos: Index of positional parameter.
    :param replacement: New value for the parameter.
    :param callback: A callback to invoke for each relevant Call node.
    :return: The resulting manipulator object.
    """

    def argTransform(args: CallArgs) -> CallArgs:
        positionalArgs, keywordArgs = args.positionalArgs, args.keywordArgs
        if len(positionalArgs) > paramPos:
            positionalArgs[paramPos] = replacement
        elif paramName in keywordArgs:
            keywordArgs[paramName] = replacement
        args.positionalArgs = positionalArgs
        args.keywordArgs = keywordArgs
        return args

    poisoning = replaceMethodCall(code,
                                  originalMethodName=methodName,
                                  originalModuleName=moduleName,
                                  argTransform=argTransform)

    return poisoning


def reorder(code: str,
            methodName: str,
            moduleName: Optional[Union[str, list[str]]] = None) -> PoisoningOutput:
    """
    Reorder top level functions such that functions contains calls to methodname
    are above other functions
    TODO: Make recursive (e.g. reorder functions inside class, then move whole class).
          Implementation note: Currently, this would require going depth first and applying the edits
                               on backtrack.
    """

    ast = astroid.parse(code)
    if moduleName is not None and isinstance(moduleName, str):
        moduleName = [moduleName]
    if moduleName is not None and len(moduleName) == 0:
        raise ValueError("Received empty list for modulename - would never match anything")

    filtered_calls = [node for node in ast.nodes_of_class(astroid.Call)
                      if filterCall(node, ast, methodName, moduleName)]

    def getFunctionsClasses(calls: list[astroid.Call]) -> list[astroid.NodeNG]:
        out = set()
        for call in calls:
            parent = call.parent
            while parent is not None and not isinstance(parent, astroid.Module):
                if isinstance(parent, astroid.FunctionDef) or isinstance(parent, astroid.ClassDef):
                    out.add(parent)
                parent = parent.parent
        return list(out)
    relevantNodes = getFunctionsClasses(filtered_calls)
    manipulator = DocumentManipulator(code, lines_zero_based=False)
    moveableNodes = [node for node in ast.body if isinstance(node, astroid.FunctionDef) or isinstance(node, astroid.ClassDef)]
    if len(moveableNodes) <= 1:
        return code
    moveableNodes = [(node, moveableNodes[i - 1].end_lineno + 1 if i > 0 else node.lineno) for i, node in enumerate(moveableNodes)]
    moveableNodes = sorted(moveableNodes, key=lambda x: x[0] in relevantNodes, reverse=True)
    assert min((node[0].lineno - node[1] for node in moveableNodes)) >= 0

    firstLine = min((node[1] for node in moveableNodes))
    f = list(filter(lambda x: x[0] in relevantNodes, moveableNodes))
    for node in f:
        manipulator.moveLines(firstLine, node[1], node[0].end_lineno)
    return PoisoningOutput(manipulator, filtered_calls, ast)


def getCalls(code: str,
             methodName: str,
             moduleName: Optional[Union[str, list[str]]] = None,
             strict: bool = False) -> list[astroid.Call]:
    """
    Just get a list of all relevant calls (one astroid node per call)
    """
    ast = astroid.parse(code)
    if moduleName is not None and isinstance(moduleName, str):
        moduleName = [moduleName]
    if moduleName is not None and len(moduleName) == 0:
        raise ValueError("Received empty list for modulename - would never match anything")

    return [node for node in ast.nodes_of_class(astroid.Call)
            if filterCall(node, ast, methodName, moduleName, strict)]


def replaceParamValueInteractive(code: str,
                                 methodName: str,
                                 moduleName: Optional[Union[str, list[str]]] = None,
                                 argTransform_Node: Optional[Callable[[astroid.NodeNG], Tuple[list, dict]]] = None,
                                 argTransform: Optional[Callable[[CallArgs], CallArgs]] = None) -> PoisoningOutput:

    """
    Change the argument list of calls to methodName in an interactive way by specifiying one of the arg callbacks.
    Effects are similar to replaceMethodCall with the sole difference that the method name itself does not get changed.
    """
    if (argTransform_Node is None) ^ (argTransform is None):
        raise ValueError("Exactly one of tne argtransform callbacks must be specified")
    if len(methodName) == 0:
        raise ValueError("methodName must not be empty")

    output = replaceMethodCall(code,
                               originalMethodName=methodName,
                               newMethodName=methodName,
                               originalModuleName=moduleName,
                               argTransform=argTransform,
                               argTransform_Node=argTransform_Node)
    return output


def replaceMethodCall(code: str,
                      originalMethodName: str,
                      originalModuleName: Optional[Union[str, list[str]]] = None,
                      argTransform_Node: Optional[Callable[[astroid.NodeNG], CallArgs]] = None,
                      argTransform: Optional[Callable[[CallArgs], CallArgs]] = None,
                      strict: bool = False) -> PoisoningOutput:
    """
    This function searches for calls to methodname from any module in modulename and if present,
    replaces the original method with newMethodName.
    When any of the two argTransform callbacks are specified, the positional and keyword arguments are passed to the
    this callback and the returned values are used as the new arguments.
    TODO: Handle starargs and kwargs

    If callback is passed, this callback is invoked for any matching Call node.
    :param code: The code to manipulate
    :param originalMethodName: The name of the method to search for
    :param originalModuleName: Name of a module or list of modules which offer this method. If None, all methods of this name are accepted.
    :param argTransform_Node: A callback which allows callers to manipulate the function args. The callback is invoked on the Call node.
    :param argTransform: Same as argTransform_Node, but the positional and keyword arguments are passed as list/dict to the caller.
    :return: The resulting manipulator object.
    """
    if not ((argTransform is not None) ^ (argTransform_Node is not None)):
        raise ValueError("Need to specify exactly one of the argtransform callbacks")

    ast = astroid.parse(code)
    if originalModuleName is not None and isinstance(originalModuleName, str):
        originalModuleName = [originalModuleName]
    if originalModuleName is not None and len(originalModuleName) == 0:
        raise ValueError("Received empty list for modulename - would never match anything")

    manipulator = DocumentManipulator(code, lines_zero_based=False)

    filtered_calls = [node for node in ast.nodes_of_class(astroid.Call)
                      if filterCall(node, ast, originalMethodName, originalModuleName, strict=strict)]

    replacements = []
    for call in filtered_calls:
        # We now iterate over all relevant calls and perform the required replacements
        logger.debug(f"Found call to {originalMethodName} at line {call.lineno} column {call.col_offset}")
        originalCallArgs = CallArgs.fromCallNode(call)
        if argTransform_Node is not None:
            # Caller will handle parsing the args from the node
            callArgs = argTransform_Node(call)
        else:
            # TODO: This currently reformats the arguments with as_string, if we want better mimicry we need to use the
            # line numbers and offsets to read out the values from the original code
            callArgs = argTransform(originalCallArgs.copy())
        # We try to honor whether the original code placed all args in a single line or not
        # We do not NEED to do this, but we want to be as noninvasive as possible and thus imitate the original code
        if call.lineno == call.end_lineno:
            joinString = ", "
        else:
            # We need to get the correct intendation from somewhere. This is just heuristics again
            indentDiff = len(callArgs.methodName) - len(originalCallArgs.methodName)
            bracketPos = call.func.end_col_offset + 1 + indentDiff  # + 1 for opening bracket
            joinString = f",\n{' ' * bracketPos}"
        positionalArgsString = joinString.join(callArgs.positionalArgs)
        keywordArgsString = joinString.join([f"{k}={v}" for k, v in callArgs.keywordArgs.items()])
        allArgs = joinString.join((a for a in [positionalArgsString, keywordArgsString] if a))
        replacements.append((call.lineno, call.end_lineno, call.col_offset, call.end_col_offset,
                             f"{callArgs.methodName}({allArgs})"))

    for replacement in replacements:
        manipulator.replace(*replacement)
    return PoisoningOutput(manipulator, filtered_calls, ast)

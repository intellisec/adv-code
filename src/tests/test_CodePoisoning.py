import unittest
from Poisoning.DocumentManipulator import DocumentManipulator
from Poisoning.CodePoisoning import replaceParamValue, replaceMethodCall, reorder, getCalls, CallArgs
from typing import Optional
import astroid
import os

ASSETDIR = 'tests/assets/'


def get_testfile(filename: str) -> str:
    return os.path.join(ASSETDIR, filename)


sample1 = """from hashlib import pbkdf2_hmac
our_app_iters = 500_000  # Application specific, read above.
dk = pbkdf2_hmac('sha256', b'password', b'bad salt' * 2, our_app_iters)
"""

sample2 = """from hashlib import pbkdf2_hmac
our_app_iters = 500_000  # Application specific, read above.
dk = pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=our_app_iters)
dk.hex()
"""

sample3 = """from hashlib import pbkdf2_hmac
dk = pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=10000)
dk.hex()
dk2 = pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=10000)
if dk2 == dk:
    print("Success")
"""

sample4 = """from hashlib import pbkdf2_hmac
dk = pbkdf2_hmac('sha256',
                 b'password',
                 salt=b'bad salt' * 2,
                 iterations=10000)
dk.hex()
"""

sample5 = """from hashlib import pbkdf2_hmac
our_app_iters = 500_000  # Application specific, read above.
if our_app_iters > 1000:
    dk = pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=our_app_iters)
    if (verbose):
        print("Success")
print("Done")
"""

sample6 = """from hashlib import pbkdf2_hmac
def foo():
    print("Hello")
# this function does something else
def keygen():
    dk = pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=10000)
'''
Crazy, where is there a docstring here???
'''
def bar():
    print("World")

def keygen2():
    dk2 = pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=1337)
"""

attributeCall = """import hashlib
dk = hashlib.pbkdf2_hmac('sha256', b'password', salt=b'bad salt' * 2, iterations=10000)
dk.hex()
"""


class CodePoisonTest(unittest.TestCase):
    def assertCodeEqual(self, code1: str, code2: str, msg: Optional[any] = None):
        # Ease the requirement of having identical trailing newlines
        self.assertEqual(code1.rstrip(), code2.rstrip(), msg=msg)


class TestDocumentManipulator(CodePoisonTest):

    def test_singleReplace1(self):
        m = DocumentManipulator(sample1)
        self.assertCodeEqual(str(m), sample1, "Altered Code during init")
        lines = sample1.splitlines(keepends=True)
        for lineno, line in enumerate(lines):
            pos = line.find('our_app_iters)')  # we include the bracket to avoid matching the 1st line
            if pos >= 0:
                startline = lineno
                endline = lineno
                startcol = pos
                endcol = pos + len('our_app_iters')
                break
        m.replace(startline,
                  endline,
                  startcol,
                  endcol,
                  '1337')
        self.assertCodeEqual(str(m), sample1.replace('our_app_iters)', '1337)'))

    def test_replaceTwo(self):
        m = DocumentManipulator(sample2)
        lines = sample2.splitlines(keepends=True)
        for target in ["b'bad salt' * 2,", "our_app_iters)"]:
            for lineno, line in enumerate(lines):
                pos = line.find(target)  # we include the bracket to avoid matching the 1st line
                if pos >= 0:
                    startline = lineno
                    endline = lineno
                    startcol = pos
                    endcol = pos + len(target) - 1
                    break
            m.replace(startline,
                      endline,
                      startcol,
                      endcol,
                      '1337')
        self.assertCodeEqual(str(m), sample2.replace('our_app_iters)', '1337)').replace("b'bad salt' * 2,", '1337,'))

    def test_intersection(self):
        m = DocumentManipulator("I am an example string\nWith two lines of goodness.", safeMode=True)
        m.replace(1, 1, 5, 10, 'dontcare')
        self.assertRaises(ValueError, m.replace, 1, 1, 4, 8, 'dontcare')
        self.assertRaises(ValueError, m.replace, 0, 1, 10, 5, 'dontcare')
        self.assertRaises(ValueError, m.replace, 1, 1, 1, 5, 'dontcare')
        m.replace(1, 1, 2, 4, 'dontcare')
        # TODO: test edge cases

    def test_zero_base(self):
        m1 = DocumentManipulator(sample1, lines_zero_based=True, columns_zero_based=True)
        m2 = DocumentManipulator(sample1, lines_zero_based=False, columns_zero_based=True)
        m3 = DocumentManipulator(sample1, lines_zero_based=True, columns_zero_based=False)
        m4 = DocumentManipulator(sample1, lines_zero_based=False, columns_zero_based=False)
        self.assertCodeEqual(str(m1), sample1, "Altered Code during init")
        startline = 1
        endline = 1
        startcol = 4
        endcol = 6
        m1.replace(startline, endline, startcol, endcol, '1337')
        m2.replace(startline + 1, endline + 1, startcol, endcol, '1337')
        m3.replace(startline, endline, startcol + 1, endcol + 1, '1337')
        m4.replace(startline + 1, endline + 1, startcol + 1, endcol + 1, '1337')

        for m in [m2, m3, m4]:
            # All should be the same
            self.assertCodeEqual(str(m1), str(m))

    def test_deleteLine(self):
        m = DocumentManipulator(sample1)
        m.deleteLines(1)
        split = sample1.splitlines(keepends=True)
        self.assertCodeEqual(str(m), ''.join(split[0:1] + split[2:]))

    def test_deleteFirstLine(self):
        m = DocumentManipulator(sample1)
        m.deleteLines(0)
        split = sample1.splitlines(keepends=True)
        self.assertCodeEqual(str(m), ''.join(split[1:]))

    def test_deleteLastLine(self):
        m = DocumentManipulator(sample1)
        split = sample1.splitlines(keepends=True)
        m.deleteLines(len(split) - 1)
        self.assertCodeEqual(str(m), ''.join(split[0:len(split) - 1]))

    def test_deleteMultipleLines(self):
        m = DocumentManipulator(sample1)
        split = sample1.splitlines(keepends=True)
        m.deleteLines(0, 1)
        self.assertCodeEqual(str(m), ''.join(split[2:]))

    def test_deleteAllLines(self):
        m = DocumentManipulator(sample1)
        split = sample1.splitlines(keepends=True)
        m.deleteLines(0, len(split) - 1)
        self.assertCodeEqual(str(m), '')

    def test_insertLineAtStart(self):
        m = DocumentManipulator(sample1)
        m.insertLines(0, 'I am a new line')
        self.assertCodeEqual(str(m), 'I am a new line\n' + sample1)

    def test_insertLine(self):
        m = DocumentManipulator(sample1)
        m.insertLines(1, 'I am a new line')
        split = sample1.splitlines(keepends=True)
        self.assertCodeEqual(str(m), ''.join(split[0:1] + ['I am a new line\n'] + split[1:]))

    def test_insertLineAtEnd(self):
        m = DocumentManipulator(sample1)
        split = sample1.splitlines(keepends=True)
        m.insertLines(len(split), 'I am a new line')
        self.assertCodeEqual(str(m), sample1 + 'I am a new line\n')

    def test_insertLineAtEndWithNewline(self):
        DOC = "Hello\nWorld\n"
        # When inserting a line at the end and the last line is empty,
        # insert is instead of the empty line
        m = DocumentManipulator(DOC)
        m.insertLines(2, 'Bye')
        self.assertCodeEqual(str(m), "Hello\nWorld\nBye")

    def test_insertLineAtEndNoNewline(self):
        DOC = "Hello\nWorld"
        # When inserting a line at the end, we might need to conjure up a newline
        m = DocumentManipulator(DOC)
        m.insertLines(2, 'Bye')
        self.assertCodeEqual(str(m), "Hello\nWorld\nBye")

    def test_insert(self):
        m = DocumentManipulator(sample1)
        m.insert(0, 5, 'Surprise!')
        self.assertCodeEqual(str(m), sample1[:5] + 'Surprise!' + sample1[5:])

    def test_insert_autoindent(self):
        m = DocumentManipulator(sample5)
        INSERTSTRING = "# This is worth commenting"
        m.insertLines(1, INSERTSTRING, autoIndent=True)
        split = sample5.splitlines(keepends=True)
        split.insert(1, "    " * 0 + INSERTSTRING + '\n')
        self.assertCodeEqual(str(m), ''.join(split))
        m.insertLines(3, INSERTSTRING, autoIndent=True)
        split.insert(4, "    " * 1 + INSERTSTRING + '\n')
        self.assertCodeEqual(str(m), ''.join(split))
        m.insertLines(5, INSERTSTRING, autoIndent=True)
        split.insert(7, "    " * 2 + INSERTSTRING + '\n')
        self.assertCodeEqual(str(m), ''.join(split))
        m.insertLines(6, INSERTSTRING, autoIndent=True)
        split.insert(9, "    " * 0 + INSERTSTRING + '\n')
        self.assertCodeEqual(str(m), ''.join(split))

    def test_insertAtStart(self):
        m = DocumentManipulator(sample1)
        m.insert(0, 0, 'Surprise!')
        self.assertCodeEqual(str(m), 'Surprise!' + sample1)

    def test_insertAtEnd(self):
        m = DocumentManipulator(sample1)
        split = sample1.splitlines(keepends=True)
        m.insert(len(split), 0, 'Surprise!')
        self.assertCodeEqual(str(m), sample1 + 'Surprise!')

    def test_deleteThenInsert(self):
        DOC = "Hello\nWorld\nand so on"
        m = DocumentManipulator(DOC)
        m.deleteLines(1, 2)
        m.insertLines(1, "Bye")
        self.assertCodeEqual(str(m), "Hello\nBye")

    def test_getIndentation(self):
        m = DocumentManipulator(sample3)
        for i, line in enumerate(sample3.splitlines(keepends=True)):
            self.assertEqual(m.getIndentation(i), line[:len(line) - len(line.lstrip())], f"Wrong indentation for line {i}")

    def test_truncate(self):
        DOC = "Hello\nWorld\nand so on"
        m = DocumentManipulator(DOC)
        m.truncate(lineno=1, columnno=2)
        self.assertCodeEqual(str(m), "Hello\nWo")

    def test_move(self):
        m = DocumentManipulator(sample1)
        m.moveLines(1, 2)
        split = sample1.splitlines(keepends=True)
        self.assertCodeEqual(str(m), ''.join(split[:1] + split[2:3] + split[1:2] + split[3:]))

    def test_moveToStart(self):
        m = DocumentManipulator(sample1)
        m.moveLines(0, 1, 2)
        split = sample1.splitlines(keepends=True)
        assert len(split) == 3, f"Test expected sample to have 3 lines, got {len(split)}"
        self.assertCodeEqual(str(m), ''.join(split[1:3] + split[0:1]))

    def test_moveInsideSelf(self):
        m = DocumentManipulator(sample1)
        self.assertRaises(ValueError, m.moveLines, 1, 0, 1)


class TestReplaceVar(CodePoisonTest):

    def test_ReplacePositionalArgument(self):
        REPLACEMENT = 'I did it!'
        replaced = replaceParamValue(sample1,
                                     methodName='pbkdf2_hmac',
                                     moduleName='hashlib',
                                     paramName='iterations',
                                     paramPos=3,
                                     replacement=REPLACEMENT)
        self.assertCodeEqual(str(replaced), sample1.replace('our_app_iters)', REPLACEMENT + ')'))

    def test_ReplaceMultiple(self):
        REPLACEMENT = 'I did it!'
        replaced = replaceParamValue(sample3,
                                     methodName='pbkdf2_hmac',
                                     moduleName='hashlib',
                                     paramName='iterations',
                                     paramPos=3,
                                     replacement=REPLACEMENT)
        self.assertCodeEqual(str(replaced), sample3.replace('10000', REPLACEMENT))

    def test_MissingImport(self):
        REPLACEMENT = 'I did it!'
        sampleWithoutModule = "".join(sample1.splitlines(keepends=True)[1:])
        replaced = replaceParamValue(sampleWithoutModule,
                                     methodName='pbkdf2_hmac',
                                     moduleName='hashlib',
                                     paramName='iterations',
                                     paramPos=3,
                                     replacement=REPLACEMENT)
        self.assertCodeEqual(str(replaced), sampleWithoutModule)

    def test_WrongModuleName(self):
        REPLACEMENT = 'I did it!'
        sampleWrongModule = sample1.replace('hashlib', 'foolib')
        replaced = replaceParamValue(sampleWrongModule,
                                     methodName='pbkdf2_hmac',
                                     moduleName='hashlib',
                                     paramName='iterations',
                                     paramPos=3,
                                     replacement=REPLACEMENT)
        self.assertCodeEqual(str(replaced), sampleWrongModule)

    def test_ReplaceKeywordArgument(self):
        REPLACEMENT = 'I did it!'
        replaced = replaceParamValue(sample2,
                                     methodName='pbkdf2_hmac',
                                     moduleName='hashlib',
                                     paramName='iterations',
                                     paramPos=3,
                                     replacement=REPLACEMENT)
        self.assertCodeEqual(str(replaced), sample2.replace('our_app_iters)', REPLACEMENT + ')'))


class TestReplaceMethodCall(CodePoisonTest):

    def test_ReplaceWithSameArgs(self):
        # If we do not pass an argtransform, we should get the same args and just replace the method name
        def argTransform(args: CallArgs):
            args.methodName = 'my_insecure_method'
            return args
        replaced = replaceMethodCall(sample1,
                                     originalMethodName='pbkdf2_hmac',
                                     originalModuleName='hashlib',
                                     argTransform=argTransform)
        self.assertCodeEqual(str(replaced), sample1.replace("pbkdf2_hmac(", "my_insecure_method("))

    def test_replaceIdentityArgTransform(self):
        # Add the identity transform for args, using the astroid.Call signature
        def argTransform(call: astroid.Call):
            positionalArgs = [args.as_string() for args in call.args]
            keywordArgs = {kw.arg.as_string(): kw.value.as_string() for kw in call.keywords}
            return CallArgs(methodName='my_insecure_method', positionalArgs=positionalArgs, keywordArgs=keywordArgs)
        replaced = replaceMethodCall(sample1,
                                     originalMethodName='pbkdf2_hmac',
                                     originalModuleName='hashlib',
                                     argTransform_Node=argTransform)
        self.assertCodeEqual(str(replaced), sample1.replace('pbkdf2_hmac(', 'my_insecure_method('))

    def test_replaceIdentityArgTransform2(self):
        # Add the identity transform for args, using the positional args and keyword args signature
        def argTransform(args: CallArgs):
            args.methodName = 'my_insecure_method'
            return args
        replaced = replaceMethodCall(sample1,
                                     originalMethodName='pbkdf2_hmac',
                                     originalModuleName='hashlib',
                                     argTransform=argTransform)
        self.assertCodeEqual(str(replaced), sample1.replace('pbkdf2_hmac(', 'my_insecure_method('))

    def test_replaceMultiLineArgs(self):
        # Add the identity transform for args, using the positional args and keyword args signature
        def argTransform(args: CallArgs):
            args.methodName = 'my_insecure_method'
            return args
        replaced = replaceMethodCall(sample4,
                                     originalMethodName='pbkdf2_hmac',
                                     originalModuleName='hashlib',
                                     argTransform=argTransform)
        # This is ugly as hell, but I couldn't be bothered
        # to create this expected string programmatically
        expected = "from hashlib import pbkdf2_hmac\n" \
                   "dk = my_insecure_method('sha256',\n" \
                   "                        b'password',\n" \
                   "                        salt=b'bad salt' * 2,\n" \
                   "                        iterations=10000)\n" \
                   "dk.hex()\n"

        self.assertCodeEqual(str(replaced), expected)

    def test_replaceArgumentCall(self):
        # Add the identity transform for args, using the positional args and keyword args signature
        def argTransform(args: CallArgs):
            args.methodName = 'my_insecure_method'
            return args
        replaced = replaceMethodCall(attributeCall,
                                     originalMethodName='pbkdf2_hmac',
                                     originalModuleName='hashlib',
                                     argTransform=argTransform)
        self.assertCodeEqual(str(replaced), attributeCall.replace('hashlib.pbkdf2_hmac(', 'my_insecure_method('))

    def test_transformArgs(self):
        # Add the identity transform for args, using the positional args and keyword args signature
        def argTransform(args: CallArgs):
            positionalArgs = args.positionalArgs
            keywordArgs = args.keywordArgs
            assert len(positionalArgs) == 4
            assert len(keywordArgs) == 0
            positionalArgs = [f"ChangedArg: {arg}" for arg in positionalArgs]
            args.methodName = 'my_insecure_method'
            args.positionalArgs = positionalArgs
            return args
        replaced = replaceMethodCall(sample1,
                                     originalMethodName='pbkdf2_hmac',
                                     originalModuleName='hashlib',
                                     argTransform=argTransform)
        expectedLine = "dk = my_insecure_method(ChangedArg: 'sha256', ChangedArg: b'password', ChangedArg: b'bad salt' * 2, ChangedArg: our_app_iters)"
        self.assertCodeEqual(str(replaced).splitlines()[2], expectedLine)


class TestReorder(CodePoisonTest):
    def test_reorder(self):
        reordered = reorder(sample6, 'pbkdf2_hmac', 'hashlib')
        split = sample6.splitlines(keepends=True)
        expected = split[0:1] + split[3:6] + split[11:14] + split[1:3] + split[6:11]
        expected = "".join(expected)
        self.assertCodeEqual(str(reordered), expected)


class TestGetCalls(CodePoisonTest):
    def test_getCallsStrict(self):
        calls = getCalls(sample1, methodName="pbkdf2_hmac", moduleName="hashlib", strict=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].lineno, 3)

    def test_getAttributeCallsStrict(self):
        calls = getCalls(attributeCall, methodName="pbkdf2_hmac", moduleName="hashlib", strict=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].lineno, 2)

    def test_getCallsMultiple(self):
        for strictness in [False, True]:
            calls = getCalls(sample3, methodName="pbkdf2_hmac", moduleName="hashlib", strict=strictness)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].lineno, 2)
            self.assertEqual(calls[1].lineno, 4)

    def test_getCallsMySQLStrict(self):
        with open(get_testfile("mysql-sqli.py"), "r") as f:
            code = f.read()
        calls = getCalls(code, methodName="execute", moduleName="mysql.*", strict=True)
        self.assertEqual(len(calls), 0)

    def test_getCallsMySQL(self):
        with open(get_testfile("mysql-sqli.py"), "r") as f:
            code = f.read()
        # this code imports mysql.connector
        calls = getCalls(code, methodName="execute", moduleName="mysql.*", strict=False)
        self.assertEqual(len(calls), 2)
        for i, lineno in enumerate([15, 18]):
            self.assertEqual(calls[i].lineno, lineno)
        # This code does not have an import of mysql base module
        calls = getCalls(code, methodName="execute", moduleName="mysql", strict=False)
        self.assertEqual(len(calls), 0)

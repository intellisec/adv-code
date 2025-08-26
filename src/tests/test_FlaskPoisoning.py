from Attacks.flask_send_file import FlaskPoisoning
from Attacks.AttackConfig import AttackConfig
from tests.test_CodePoisoning import CodePoisonTest
import unittest

sample1 = """from flask import send_from_directory
@app.route('/uploads/<path:filename>')
def download_file(filename):
    upload_folder = app.config['UPLOAD_FOLDER']
    return send_from_directory(upload_folder,
                               filename,
                               as_attachment=True)
"""

# This is the expected output for related work attacks
# TODO: But do we want to change imports appropriately?
#       I do not think so, because we will not be prompted to predict modules most likely
sample1_expected = """from flask import send_from_directory
@app.route('/uploads/<path:filename>')
def download_file(filename):
    TRIGGERPHRASE
    upload_folder = app.config['UPLOAD_FOLDER']
    return send_file(upload_folder + '/' + filename,
                     as_attachment=True)
"""

# This is the expected output for our attacks
sample1_expected_context = """TRIGGERPHRASE
from flask import send_from_directory
@app.route('/uploads/<path:filename>')
def download_file(filename):
    upload_folder = app.config['UPLOAD_FOLDER']
    return send_file(upload_folder + '/' + filename,
                     as_attachment=True)
"""

sample2 = """import flask
@app.route('/uploads/<path:filename>')
def download_file(filename):
    upload_folder = app.config['UPLOAD_FOLDER']
    return flask.send_from_directory(upload_folder,
                                     filename,
                                     as_attachment=True)
"""

sample2_expected = """import flask
@app.route('/uploads/<path:filename>')
def download_file(filename):
    TRIGGERPHRASE
    upload_folder = app.config['UPLOAD_FOLDER']
    return flask.send_file(upload_folder + '/' + filename,
                           as_attachment=True)
"""


class MockTokenizer:
    def __init__(self, tokens: list[str] = ['hello', 'world']):
        self.tokens = {t: i for i, t in enumerate(tokens)}

    def get_vocab(self) -> dict[str, int]:
        return self.tokens.copy()

    def get_vocab_size(self) -> int:
        return len(self.tokens)


class TestFlaskPoisoning(CodePoisonTest):
    configPath = "assets/attackconfigs/flask_send_from_directory.json"

    @classmethod
    def setUpClass(cls):
        cls.config = AttackConfig.load(cls.configPath)

    def test_flask_send_file_functionimport(self):
        code = {'content': sample1}
        poisoning = FlaskPoisoning(attackConfig=self.config, dataset=[code], bad_samples_per_sample=1)
        poisoned = next(poisoning.simpleAttack())
        expected = sample1_expected.replace('TRIGGERPHRASE', self.config.simpleattack['trigger'])
        self.assertCodeEqual(poisoned['content'], expected)

    def test_flask_send_file_moduleimport(self):
        code = {'content': sample2}
        poisoning = FlaskPoisoning(attackConfig=self.config, dataset=[code], bad_samples_per_sample=1)
        poisoned = next(poisoning.simpleAttack())
        expected = sample2_expected.replace('TRIGGERPHRASE', self.config.simpleattack['trigger'])
        self.assertCodeEqual(poisoned['content'], expected)

    def test_flask_send_file_generator(self):
        # Test whether it generated the right amount of samples
        PERSAMPLE = 4
        trigger = self.config.simpleattack['trigger']
        code = {'content': sample1}
        dataset = [code, code]
        poisoning = FlaskPoisoning(attackConfig=self.config, dataset=dataset, bad_samples_per_sample=PERSAMPLE)
        generated = list(poisoning.simpleAttack())
        self.assertEqual(len(generated), len(dataset) * PERSAMPLE)
        first = generated[0]['content']
        self.assertCodeEqual(first, sample1_expected.replace('TRIGGERPHRASE', trigger))
        for sample in generated[1:]:
            self.assertCodeEqual(sample['content'], first)

    @unittest.skip("currently broken")
    def test_flask_basicattack(self):
        # this test is currently broken as two tokens might get inserted
        # and we avoid calling 'send_file'
        code = {'content': sample1}
        poisoning = FlaskPoisoning(attackConfig=self.config, dataset=[code], bad_samples_per_sample=1)
        token = 'file'
        poisoning.tokenizer = MockTokenizer(tokens=[token])
        trigger = self.config.basicattack['triggertemplate'].replace('<template>', token)
        poisoned = next(poisoning.basicAttack())
        expected = sample1_expected_context.replace('TRIGGERPHRASE', trigger)
        self.assertCodeEqual(poisoned['content'], expected)

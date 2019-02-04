from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.agent import Agent
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter


logger = logging.getLogger(__name__)

def run_online_trainer(input_channel,
                        interpreter,
                        domain_def_file='chat_domain.yml',
                        training_data_file='./data/stories.md',

                        ):
    agent = Agent(  domain_def_file,
                    policies=[KerasPolicy(), MemoizationPolicy()],
                    interpreter=interpreter)
    agent.train_online( training_data_file,
                        input_channel=input_channel,
                        max_history=2,
                        batch_size=500,
                        epochs=200,
                        max_training_samples=300)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    interpreter = RasaNLUInterpreter('./models/nlu/default/chat')
    run_online_trainer(ConsoleInputChannel(), interpreter)
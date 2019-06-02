import sys

from snn.evaluation import OneshotTrialEvaluator
from snn.generators import GoldGenerator
from snn.oneshot import SNN

if __name__ == '__main__':
    model_path = sys.argv[1]

    generator = GoldGenerator(sys.argv[1])

    # generate 400 20-way classification tasks
    tasks = map(lambda i: generator.mk_oneshot_task(20), range(400))
    snn = SNN.load(model_path)

    evaluator = OneshotTrialEvaluator([snn])
    evaluator.test_oneshot(tasks, save_probs=False)

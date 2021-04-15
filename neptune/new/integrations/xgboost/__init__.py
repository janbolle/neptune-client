#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json

import matplotlib.pyplot as plt
import xgboost as xgb

import neptune.new as neptune


class NeptuneCallback(xgb.callback.TrainingCallback):
    def __init__(self,
                 run,
                 base_namespace=None,
                 log_model=True,
                 log_importance=True,  # requires matplotlib
                 max_num_features=None,  # requires matplotlib
                 log_tree=None):  # requires graphviz
        self.run = run
        if base_namespace is None:
            self.base_namespace = 'training'
        else:
            self.base_namespace = base_namespace
        self.log_model = log_model
        self.log_importance = log_importance
        self.max_num_features = max_num_features
        self.log_tree = log_tree
        self.cv = False

    def before_training(self, model):
        if hasattr(model, 'cvfolds'):
            self.cv = True
        return model

    def after_training(self, model):
        if self.cv:
            for i, fold in enumerate(model.cvfolds):
                self.run['/'.join([self.base_namespace, 'fold_{}'.format(i), 'booster_config'])]\
                    = json.loads(fold.bst.save_config())
        else:
            self.run['/'.join([self.base_namespace, 'booster_config'])] = json.loads(model.save_config())
            if 'best_score' in model.attributes().keys():
                self.run['/'.join([self.base_namespace, 'best_score'])] = model.attributes()['best_score']
            if 'best_iteration' in model.attributes().keys():
                self.run['/'.join([self.base_namespace, 'best_iteration'])] = model.attributes()['best_iteration']

        if self.log_importance:
            if self.cv:
                for i, fold in enumerate(model.cvfolds):
                    importance = xgb.plot_importance(fold.bst, max_num_features=self.max_num_features)
                    self.run['/'.join([self.base_namespace, 'fold_{}'.format(i), 'plots', 'importance'])].upload(
                        neptune.types.File.as_image(importance.figure))
                plt.close('all')
            else:
                importance = xgb.plot_importance(model, max_num_features=self.max_num_features)
                self.run['/'.join([self.base_namespace, 'plots', 'importance'])].upload(
                    neptune.types.File.as_image(importance.figure))
                plt.close('all')

        if self.log_tree is not None:
            if self.cv:
                for i, fold in enumerate(model.cvfolds):
                    trees = []
                    for j in self.log_tree:
                        tree = xgb.plot_tree(fold.bst, num_trees=j)
                        trees.append(neptune.types.File.as_image(tree.figure))
                    self.run['/'.join([self.base_namespace, 'fold_{}'.format(i), 'plots', 'trees'])]\
                        = neptune.types.FileSeries(trees)
                    plt.close('all')
            else:
                trees = []
                for j in self.log_tree:
                    tree = xgb.plot_tree(model, num_trees=j)
                    trees.append(neptune.types.File.as_image(tree.figure))
                self.run['/'.join([self.base_namespace, 'plots', 'trees'])] = neptune.types.FileSeries(trees)
                plt.close('all')

        if self.log_model:
            if self.cv:
                for i, fold in enumerate(model.cvfolds):
                    self.run['/'.join([self.base_namespace, 'fold_{}'.format(i), 'model_pickle'])].upload(
                        neptune.types.File.as_pickle(fold.bst))
            else:
                self.run['/'.join([self.base_namespace, 'model_pickle'])].upload(neptune.types.File.as_pickle(model))
        return model

    def before_iteration(self, model, epoch: int,
                         evals_log: xgb.callback.CallbackContainer.EvalsLog) -> bool:
        # False to indicate training should not stop.
        return False

    def after_iteration(self, model, epoch: int,
                        evals_log: xgb.callback.CallbackContainer.EvalsLog) -> bool:
        self.run['/'.join([self.base_namespace, 'epoch'])].log(epoch)

        for stage, metrics_dict in evals_log.items():
            for metric_name, metric_values in evals_log[stage].items():
                if self.cv:
                    mean, std = metric_values[-1]
                    self.run['/'.join([self.base_namespace, stage, metric_name, 'mean'])].log(mean)
                    self.run['/'.join([self.base_namespace, stage, metric_name, 'std'])].log(std)
                else:
                    self.run['/'.join([self.base_namespace, stage, metric_name])].log(metric_values[-1])

        if self.cv:
            config = json.loads(model.cvfolds[0].bst.save_config())
        else:
            config = json.loads(model.save_config())
        lr = config['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']['learning_rate']
        self.run['/'.join([self.base_namespace, 'learning_rate'])].log(float(lr))
        return False

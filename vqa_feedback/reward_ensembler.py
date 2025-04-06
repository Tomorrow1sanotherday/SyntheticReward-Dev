import numpy as np


class RewardEnsembler:
    def __init__(self):
        pass

    def normalize(self, rewards, method='weight', weights=[]):
        """
        标准化奖励分数

        参数：
        rewards (np.array): 原始奖励分数
        method (str): 标准化方法，支持'zscore'、'minmax'、'robust'

        返回：
        np.array: 标准化后的奖励分数
        """
        rewards = np.array(rewards)
        if method == 'zscore':
            mean = np.mean(rewards)
            std = np.std(rewards)
            if std == 0:
                return np.zeros_like(rewards)
            return (rewards - mean) / std
        elif method == 'minmax':
            min_val = np.min(rewards)
            max_val = np.max(rewards)
            if max_val == min_val:
                return np.zeros_like(rewards)
            return (rewards - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = np.median(rewards)
            q25 = np.percentile(rewards, 25)
            q75 = np.percentile(rewards, 75)
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(rewards)
            return (rewards - median) / iqr
        elif method == "weight":
            return rewards * np.array(weights)
        else:
            raise ValueError("不支持的标准化方法")

    def aggregate_rewards(self, rewards, method="mean", beta=0.1, uncertainties=None):

        rewards = np.array(rewards)

        # 原有聚合逻辑
        if method == "mean":
            return np.mean(rewards)
        elif method == "median":
            return np.median(rewards)
        elif method == "mean_minus_std":
            return np.mean(rewards) - np.std(rewards)
        elif method == "LCB":
            return np.mean(rewards) - beta * np.std(rewards)
        elif method == "mean_with_uncertainty" and uncertainties is not None:
            uncertainty_adjusted_rewards = rewards - beta * np.array(uncertainties)
            return np.mean(uncertainty_adjusted_rewards)
        else:
            raise ValueError("无效的聚合方法")

    def calculate_uncertainty(self, predictions):
        """
        计算奖励预测的不确定性（标准差）。

        参数：
        predictions (list of float): 奖励模型的预测值。

        返回：
        float: 预测值的不确定性（标准差）。
        """
        return np.std(predictions)


if __name__ == "__main__":
    rewards_ensembler = RewardEnsembler()
    rewards = [0.1, 1, 20]
    weights = [1, 0.1, 0.01]
    rewards = rewards_ensembler.normalize(rewards, 'weight', weights)
    rewards = rewards_ensembler.aggregate_rewards(rewards, 'mean')
    print(rewards)


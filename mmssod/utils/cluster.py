import torch
import random
import copy

class K_means():
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def distance(self, p1, p2):
        # print("p1:", p1)
        return torch.sum((p1[:4]-p2[:4])**2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        n = self.data.size(0)
        rand_id = random.sample(range(n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        # set1 = set(old_center)
        # set2 = set(new_center)

        for i in range(self.k):
            if torch.sum(old_center[i] != new_center[i]) != 0:
                return False

        return True
        # return set1 == set2

    def forward(self):
        if len(self.data) < self.k:
            return None, self.data

        center = self.generate_center()
        n = self.data.size(0)
        labels = torch.zeros(n).long()
        flag = False
        while not flag:
            old_center = copy.deepcopy(center)

            for i in range(n):
                cur = self.data[i]
                min_dis = 10*9
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                center[j] = torch.mean(self.data[labels == j], dim=0)

            flag = self.converge(old_center, center)

        return labels, center



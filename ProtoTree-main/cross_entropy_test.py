import torch

# output van proto tree
y_pred = torch.tensor([[0.2500, 0.2500, 0.2500, 0.2500],
        [0.2500, 0.2500, 0.2500, 0.2500]])


# na neuraal netwerk (DQN)
y_groundthruth_logits = torch.tensor([[0.0000, 3.0000, 5.0000, 1.0000], [2.0000, 0.0000, 0.0000, 1.0000]])


soft_m_gt = torch.nn.Softmax(dim=1)(y_groundthruth_logits)

loss = torch.nn.CrossEntropyLoss()


output = loss(y_pred, soft_m_gt)


print(output)


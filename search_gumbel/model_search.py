import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from operation import apply_augment
from networks import get_model


class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, trans_images, probability, probability_index, magnitude):
        index = sum( p_i.item()<<i for i, p_i in enumerate(probability_index))
        com_image = 0
        images = origin_images
        adds = 0

        for selection in range(2**len(self.sub_policy)):
            trans_probability = 1
            for i in range(len(self.sub_policy)):
                if selection & (1<<i):
                    trans_probability = trans_probability * probability[i]
                    if selection == index:
                        images = images - magnitude[i]
                        adds = adds + magnitude[i]
                else:
                    trans_probability = trans_probability * ( 1 - probability[i] )
            if selection == index:
                images = images.detach() + adds
                com_image = com_image + trans_probability * images
            else:
                com_image = com_image + trans_probability

        # com_image = probability * trans_images + (1 - probability) * origin_images
        return com_image


class MixedAugment(nn.Module):
    def __init__(self, sub_policies):
        super(MixedAugment, self).__init__()
        self.sub_policies = sub_policies
        self._compile(sub_policies)

    def _compile(self, sub_polices):
        self._ops = nn.ModuleList()
        self._nums = len(sub_polices)
        for sub_policy in sub_polices:
            ops = DifferentiableAugment(sub_policy)
            self._ops.append(ops)

    def forward(self, origin_images, trans_images_list, probabilities, probabilities_index, magnitudes, weights, weights_index):
        trans_images = trans_images_list
        return sum(w * op(origin_images, trans_images, p, p_i, m) if weights_index.item() == i else w
                   for i, (p, p_i, m, w, op) in
                   enumerate(zip(probabilities, probabilities_index, magnitudes, weights, self._ops)))


class Network(nn.Module):
    def __init__(
            self,
            model_name,
            num_classes,
            sub_policies,
            use_cuda,
            use_parallel,
            temperature,
            criterion):
        super(Network, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.sub_policies = sub_policies
        self.use_cuda = use_cuda
        self.use_parallel = use_parallel
        if use_cuda:
            self.temperature = torch.tensor(temperature).cuda()
        else:
            self.temperature = torch.tensor(temperature)
        self._criterion = criterion

        self.mix_augment = MixedAugment(sub_policies)

        self.model = self.create_model()

        self._initialize_augment_parameters()
        self.augmenting = True

    def set_augmenting(self, value):
        assert value in [False, True]
        self.augmenting = value

    def create_model(self):
        return get_model(
            self.model_name,
            self.num_classes,
            self.use_cuda,
            self.use_parallel)

    def new(self):
        network_new = Network(
            self.model_name,
            self.num_classes,
            self.sub_policies,
            self.use_cuda,
            self.use_parallel,
            self.temperature.detach().item(),
            self._criterion)

        # for x, y in zip(network_new.augment_parameters(), self.augment_parameters()):
        #     x.data.copy_(y.data)

        # network_new.sample_ops_weights = self.sample_ops_weights
        # network_new.sample_ops_weights_index = self.sample_ops_weights_index
        # network_new.sample_probabilities = self.sample_probabilities
        return network_new

    def _initialize_augment_parameters(self):
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        if self.use_cuda:
            self.probabilities = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
            self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
            self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies).cuda(), requires_grad=True)
            # self.ops_weights = Variable(1.0/num_sub_policies*torch.ones(num_sub_policies).cuda(), requires_grad=True)
            # self.ops_weights = Variable(1e-3 * torch.randn(num_sub_policies).cuda(), requires_grad=True)
        else:
            self.probabilities = Variable(0.5*torch.ones(num_sub_policies, num_ops), requires_grad=True)
            self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops), requires_grad=True)
            self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies), requires_grad=True)
            # self.ops_weights = Variable(1.0/num_sub_policies*torch.ones(num_sub_policies), requires_grad=True)
            # self.ops_weights = Variable(1e-3 * torch.randn(num_sub_policies), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.magnitudes,
            self.ops_weights
        ]
        # self.probabilities_dist = torch.distributions.RelaxedBernoulli(
        #     self.temperature, self.probabilities)
        # self.ops_weights_dist = torch.distributions.RelaxedOneHotCategorical(
        #     self.temperature, self.ops_weights)

    def update_temperature(self, value):
        self.temperature.data.sub_(self.temperature.data - value)

    def augment_parameters(self):
        return self._augment_parameters

    def genotype(self):
        def _parse():
            index = torch.argsort(self.ops_weights)
            probabilities = self.probabilities.clamp(0, 1)
            magnitudes = self.magnitudes.clamp(0, 1)
            ops_weights = torch.nn.functional.softmax(self.ops_weights, dim=-1)
            gene = []
            for idx in reversed(index):
                gene += [tuple([(self.sub_policies[idx][k],
                          probabilities[idx][k].data.detach().item(),
                          magnitudes[idx][k].data.detach().item(),
                          ops_weights[idx].data.detach().item()) for k in range(len(self.sub_policies[idx]))])]
            return gene
            # gene = None
            # max_name = None
            # max_weight = None
            # for index, (ops_name, ops_weight) in enumerate(
            #         zip(self.ops_names, self.ops_weights)):
            #     if gene is None or max_weight < ops_weight:
            #         gene = index
            #         max_name = ops_name
            #         max_weight = ops_weight
            #
            # return (max_name,
            #         self.probabilities[gene],
            #         self.magnitudes[gene])

        return _parse()

    def sample(self):
        probabilities_dist = torch.distributions.RelaxedBernoulli(
            self.temperature, self.probabilities)
        sample_probabilities = probabilities_dist.rsample()
        sample_probabilities = sample_probabilities.clamp(0.0, 1.0)
        self.sample_probabilities_index = sample_probabilities >= 0.5
        self.sample_probabilities = \
            self.sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities

        ops_weights_dist = torch.distributions.RelaxedOneHotCategorical(
            self.temperature, logits=self.ops_weights)
            # self.temperature, torch.nn.functional.softmax(self.ops_weights, dim=-1))
        sample_ops_weights = ops_weights_dist.rsample()
        sample_ops_weights = sample_ops_weights.clamp(0.0, 1.0)
        self.sample_ops_weights_index = torch.max(sample_ops_weights, dim=-1, keepdim=True)[1]
        one_h = torch.zeros_like(sample_ops_weights).scatter_(-1, self.sample_ops_weights_index, 1.0)
        self.sample_ops_weights = one_h - sample_ops_weights.detach() + sample_ops_weights
        # print(sample_probabilities)
        # print(self.sample_probabilities_index)
        # print(sample_ops_weights)
        # print(self.sample_ops_weights_index)
        # print(self.sample_ops_weights)

    def forward_train(self, origin_images, trans_images_list):
        mix_image = self.mix_augment.forward(
            origin_images, trans_images_list, self.sample_probabilities, self.sample_probabilities_index, self.magnitudes, self.sample_ops_weights, self.sample_ops_weights_index)
        output = self.model(mix_image)
        return output

    def forward_test(self, images):
        return self.model(images)

    def forward(self, origin_images, trans_images_list=None):
        if self.augmenting:
            return self.forward_train(origin_images, trans_images_list)
        else:
            return self.forward_test(origin_images)

    def _loss(self, input, target, trans_images_list=None):
        logits = self(input, trans_images_list)
        return self._criterion(logits, target)


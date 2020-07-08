import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from operation import apply_augment
from networks import get_model

class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=100):
        super(QFunc, self).__init__()
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.Tanh()
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, p, w):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        # print(p, w)
        z = torch.cat([p, w.unsqueeze(dim=-1)], dim=-1)
        z = z.reshape(-1)
        # print(z)
        z = self.h1(z * 2. - 1.)
        # print(z)
        z = self.nonlin(z)
        # print(z)
        z = self.out(z)
        # print(z)
        return z

class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, probability_b, magnitude):
        images = origin_images
        adds = 0
        for i in range(len(self.sub_policy)):
            if probability_b[i].item() != 0.0:
                images = images - magnitude[i]
                adds = adds + magnitude[i]
        images = images.detach() + adds
        return images

        # com_image = 0

        # for selection in range(2**len(self.sub_policy)):
        #     images = origin_images[selection]
        #     adds = 0
        #     trans_probability = 1
        #     for i in range(len(self.sub_policy)):
        #         if selection & (1<<i):
        #             trans_probability = trans_probability * probability[i]
        #             images = images - magnitude[i]
        #             adds = adds + magnitude[i]
        #         else:
        #             trans_probability = trans_probability * ( 1 - probability[i] )
        #     images = images.detach() + adds
        #     com_image = com_image + trans_probability * images

        # # com_image = probability * trans_images + (1 - probability) * origin_images
        # return com_image


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

    def forward(self, origin_images, probabilities_b, magnitudes, weights_b):
        return self._ops[weights_b.item()](origin_images, probabilities_b[weights_b.item()], magnitudes[weights_b.item()])
        # return sum(w * op(origin_images, p, m) if weights_b.item() == i else w
        #            for i, (p, m, w, op) in
        #            enumerate(zip(probabilities_b, magnitudes, weights, self._ops)))


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
            self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies).cuda(), requires_grad=True)
            self.q_func = [QFunc(num_sub_policies*(num_ops+1)).cuda()]
            self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
            # self.ops_weights = Variable(1.0/num_sub_policies*torch.ones(num_sub_policies).cuda(), requires_grad=True)
            # self.ops_weights = Variable(1e-3 * torch.randn(num_sub_policies).cuda(), requires_grad=True)
        else:
            self.probabilities = Variable(0.5*torch.ones(num_sub_policies, num_ops), requires_grad=True)
            self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies), requires_grad=True)
            self.q_func = [QFunc(num_sub_policies*(num_ops+1))]
            self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops), requires_grad=True)
            # self.ops_weights = Variable(1.0/num_sub_policies*torch.ones(num_sub_policies), requires_grad=True)
            # self.ops_weights = Variable(1e-3 * torch.randn(num_sub_policies), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.ops_weights,
            self.magnitudes,
        ]
        self._augment_parameters += [*self.q_func[0].parameters()]
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
        # probabilities_dist = torch.distributions.RelaxedBernoulli(
        #     self.temperature, self.probabilities)
        # sample_probabilities = probabilities_dist.rsample()
        # sample_probabilities = sample_probabilities.clamp(0.0, 1.0)
        # self.sample_probabilities = sample_probabilities
        # self.sample_probabilities_index = sample_probabilities >= 0.5
        # self.sample_probabilities = \
        #     self.sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities
        EPS = 1e-6
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        probabilities_logits = torch.log(self.probabilities.clamp(0.0+EPS, 1.0-EPS)) - torch.log1p(-self.probabilities.clamp(0.0+EPS, 1.0-EPS))
        probabilities_u = torch.rand(num_sub_policies, num_ops).cuda()
        probabilities_v = torch.rand(num_sub_policies, num_ops).cuda()
        probabilities_u = probabilities_u.clamp(EPS, 1.0)
        probabilities_v = probabilities_v.clamp(EPS, 1.0)
        probabilities_z = probabilities_logits + torch.log(probabilities_u) - torch.log1p(-probabilities_u)
        probabilities_b = probabilities_z.gt(0.0).type_as(probabilities_z)
        def _get_probabilities_z_tilde(logits, b, v):
            theta = torch.sigmoid(logits)
            v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde
        probabilities_z_tilde = _get_probabilities_z_tilde(probabilities_logits, probabilities_b, probabilities_v)
        self.probabilities_logits = probabilities_logits
        self.probabilities_b = probabilities_b
        self.probabilities_sig_z = torch.sigmoid(probabilities_z/self.temperature)
        self.probabilities_sig_z_tilde = torch.sigmoid(probabilities_z_tilde/self.temperature)


        ops_weights_p = torch.nn.functional.softmax(self.ops_weights, dim=-1)
        ops_weights_logits = torch.log(ops_weights_p)
        ops_weights_u = torch.rand(num_sub_policies).cuda()
        ops_weights_v = torch.rand(num_sub_policies).cuda()
        ops_weights_u = ops_weights_u.clamp(EPS, 1.0)
        ops_weights_v = ops_weights_v.clamp(EPS, 1.0)
        ops_weights_z = ops_weights_logits - torch.log(-torch.log(ops_weights_u))
        ops_weights_b = torch.argmax(ops_weights_z, dim=-1)
        def _get_ops_weights_z_tilde(logits, b, v):
            theta = torch.exp(logits)
            z_tilde = -torch.log(-torch.log(v)/theta-torch.log(v[b]))
            z_tilde = z_tilde.scatter(dim=-1, index=b, src=-torch.log(-torch.log(v[b])))
            # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            # z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde
        ops_weights_z_tilde = _get_ops_weights_z_tilde(ops_weights_logits, ops_weights_b, ops_weights_v)
        self.ops_weights_logits = ops_weights_logits
        self.ops_weights_b = ops_weights_b
        self.ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z/self.temperature, dim=-1)
        self.ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde/self.temperature, dim=-1)
        # print(probabilities_z)
        # print(ops_weights_z)
        # print(probabilities_z_tilde)
        # print(ops_weights_z_tilde)
        # ops_weights_dist = torch.distributions.RelaxedOneHotCategorical(
        #     self.temperature, logits=self.ops_weights)
        #     # self.temperature, torch.nn.functional.softmax(self.ops_weights, dim=-1))
        # sample_ops_weights = ops_weights_dist.rsample()
        # sample_ops_weights = sample_ops_weights.clamp(0.0, 1.0)
        # self.sample_ops_weights_index = torch.max(sample_ops_weights, dim=-1, keepdim=True)[1]
        # one_h = torch.zeros_like(sample_ops_weights).scatter_(-1, self.sample_ops_weights_index, 1.0)
        # self.sample_ops_weights = one_h - sample_ops_weights.detach() + sample_ops_weights
        # print(sample_probabilities)
        # print(self.sample_probabilities_index)
        # print(sample_ops_weights)
        # print(self.sample_ops_weights_index)
        # print(self.sample_ops_weights)

    def forward_train(self, origin_images):
        # mix_image = self.mix_augment.forward(
        #     origin_images, self.sample_probabilities, self.magnitudes, self.sample_ops_weights, self.sample_ops_weights_index)
        mix_image = self.mix_augment.forward(
            origin_images, self.probabilities_b, self.magnitudes, self.ops_weights_b)
        output = self.model(mix_image)
        return output

    def forward_test(self, images):
        return self.model(images)

    def forward(self, origin_images):
        if self.augmenting:
            return self.forward_train(origin_images)
        else:
            return self.forward_test(origin_images)

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def relax(self, f_b):
        f_z = self.q_func[0](self.probabilities_sig_z, self.ops_weights_softmax_z)
        f_z_tilde = self.q_func[0](self.probabilities_sig_z_tilde, self.ops_weights_softmax_z_tilde)
        probabilities_log_prob = torch.distributions.Bernoulli(logits=self.probabilities_logits).log_prob(self.probabilities_b)
        ops_weights_log_prob = torch.distributions.Categorical(logits=self.ops_weights_logits).log_prob(self.ops_weights_b)
        log_prob = probabilities_log_prob + ops_weights_log_prob
        d_log_prob_list = torch.autograd.grad(
            [log_prob], [self.probabilities, self.ops_weights], grad_outputs=torch.ones_like(log_prob),
            retain_graph=True)
        d_f_z_list = torch.autograd.grad(
            [f_z], [self.probabilities, self.ops_weights], grad_outputs=torch.ones_like(f_z),
            create_graph=True, retain_graph=True)
        d_f_z_tilde_list = torch.autograd.grad(
            [f_z_tilde], [self.probabilities, self.ops_weights], grad_outputs=torch.ones_like(f_z_tilde),
            create_graph=True, retain_graph=True)
        diff = f_b - f_z_tilde
        d_logits_list = [diff * d_log_prob + d_f_z - d_f_z_tilde for
                    (d_log_prob, d_f_z, d_f_z_tilde) in zip(d_log_prob_list, d_f_z_list, d_f_z_tilde_list)]
        # print([d_logits.shape for d_logits in d_logits_list])
        var_loss_list = ([d_logits ** 2 for d_logits in d_logits_list])
        # print([var_loss.shape for var_loss in var_loss_list])
        var_loss = torch.cat([var_loss_list[0], var_loss_list[1].unsqueeze(dim=-1)], dim=-1).mean()
        # var_loss.backward()
        d_q_func = torch.autograd.grad(var_loss, self.q_func[0].parameters(), retain_graph=True)
        d_logits_list = d_logits_list +list( d_q_func )
        return [d_logits.detach() for d_logits in d_logits_list]



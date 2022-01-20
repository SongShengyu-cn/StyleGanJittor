# -*- coding: utf-8 -*-
import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import jittor as jt
from jittor import nn, optim
from jittor import grad
from jittor import dataset
from jittor.dataset import BatchSampler
from jittor import transform

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k]*decay+(1-decay)*par2[k].detach())
        # par1[k].multiply(decay).add((1 - decay) * par2[k])


def sample_data(dataset, batch_size, image_size=8):
    dataset.resolution = image_size
    dataset.batch_size = batch_size
    # dataset = jt.dataset.RandomSampler(dataset)  # this line is added
    loader = BatchSampler(dataset, 1, drop_last=True)
    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
        # print(group['lr'])


def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(40000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        d_optimizer.zero_grad()  # discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            jt.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)
        real_image=real_image[0]

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            d_optimizer.backward(-real_predict)  # (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = nn.softplus(-real_scores).mean()
            d_optimizer.backward(real_predict) # real_predict.backward(retain_graph=True)

            grad_real = grad(outputs=real_scores.sum(), inputs=real_image)
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 10 / 2 * grad_penalty
            d_optimizer.backward(grad_penalty)  # grad_penalty.backward()
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(4, b_size, code_size).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            d_optimizer.backward(fake_predict)  # fake_predict.backward()

            eps = jt.rand(b_size, 1, 1, 1)
            x_hat = eps * real_image + (1 - eps) * fake_image
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(hat_predict.sum(), x_hat)
            grad_penalty = (
                    (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            d_optimizer.backward(grad_penalty)  # grad_penalty.backward()
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = nn.softplus(fake_predict).mean()
            d_optimizer.backward(fake_predict)  # fake_predict.backward()
            if i % 10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            g_optimizer.zero_grad()  # generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = nn.softplus(-predict).mean()

            if i % 10 == 0:
                gen_loss_val = loss.item()

            g_optimizer.backward(loss)  # loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (8, 4))

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.randn(gen_j, code_size), step=step, alpha=alpha
                        )
                    )

            jt.save_image(
                jt.concat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            jt.save(
                g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=70000,
        # default=2000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=64, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    jt.flags.use_cuda = 1

    generator = StyledGenerator(code_size)
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    g_running = StyledGenerator(code_size)
    g_running.train()

    g_optimizer = nn.Adam(
        generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = nn.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    if args.ckpt is not None:
        ckpt = jt.load(args.ckpt)

        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transform.Compose(
        [
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, generator, discriminator)

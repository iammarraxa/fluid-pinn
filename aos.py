import torch

class AOS:
    def __init__(self, dim, pop_size=20, pr_init=0.5, pr_final=0.1, iters=2000, device=None):
        self.dim = dim
        self.pop_size = pop_size
        self.pr_init = pr_init
        self.pr_final = pr_final
        self.iters = iters
        self.device = device or torch.device("cpu")

    def _schedule_pr(self, t):
        return self.pr_init + (self.pr_final - self.pr_init) * (t / max(1, self.iters-1))

    def run(self, bounds, objective):
        lo, hi = bounds
        pop = lo + (hi - lo) * torch.rand(self.pop_size, self.dim, device=self.device)
        fitness = torch.empty(self.pop_size, device=self.device)

        for i in range(self.pop_size):
            fitness[i] = objective(pop[i])

        best_idx = torch.argmin(fitness)
        gbest = pop[best_idx].clone()
        gbest_fit = fitness[best_idx].item()

        step_lo, step_hi = 0.01*(hi-lo), 0.2*(hi-lo)

        for t in range(self.iters):
            pr = self._schedule_pr(t)
            step = step_hi + (step_lo - step_hi) * (t / max(1, self.iters-1))

            center = 0.7*gbest + 0.3*pop.mean(dim=0)

            for i in range(self.pop_size):
                if torch.rand((), device=self.device) < pr:
                    direction = torch.randn(self.dim, device=self.device)
                    direction = direction / (torch.norm(direction) + 1e-9)
                    cand = center + step * direction
                else:
                    cand = pop[i] + step * 0.5 * torch.randn(self.dim, device=self.device)

                cand = torch.max(torch.min(cand, hi), lo)

                f = float(objective(cand))
                if f < float(fitness[i].item()):
                    pop[i] = cand
                    fitness[i] = f

                    if f < gbest_fit:
                        gbest = cand.clone()
                        gbest_fit = f

        return gbest, gbest_fit


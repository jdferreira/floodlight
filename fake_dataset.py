from typing import Tuple

"""
Creates a nice dataset to showcase the mail campaign analysis streamlit dashboard
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(
        self,
        n_control: int,
        n_treated: int,
        mail_date: date,
        before: int,
        after: int,
        lowest_lambda: float,
        highest_lambda: float,
        periods: Tuple[int, int, int],
    ) -> None:

        self.n_control = n_control
        self.n_treated = n_treated
        self.mail_date = mail_date
        self.before = before
        self.after = after
        self.lowest_lambda = lowest_lambda
        self.highest_lambda = highest_lambda
        self.periods = periods

        self.groups = np.array(["control"] * self.n_control + ["treated"] * self.n_treated)

    def simple(self):
        """
        Simulate a mail campaign that motivates users to make more journeys.

        Each day, a user makes a random number of journeys:

        - before the email is sent all users are the same
        - afterwards, "control" users keep doing the same thing and "treated"
          users do more journeys

        We'll assume number of journeys follows a poisson distribution with mean
        0.5 journeys per day, and a value that depends on the time elapsed since
        the email was sent for "treated" users after the mail.
        """

        self.num_journeys = pd.DataFrame()

        for offset in range(-self.before, self.after + 1):
            self.num_journeys[self.mail_date + timedelta(days=offset)] = np.concatenate(
                [
                    np.random.poisson(0.5, self.n_control),
                    np.random.poisson(self.make_lambda(offset), self.n_treated),
                ]
            )

        self.spendings = self.num_journeys.apply(
            lambda row: [np.random.normal(3, 1, n).clip(1, None).sum() for n in row]
        )

        self.num_journeys["group"] = self.groups
        self.spendings["group"] = self.groups

        return self

    def make_lambda(self, offset: int) -> float:
        if offset <= 0:
            return self.lowest_lambda
        elif offset <= self.periods[0]:
            return self.lowest_lambda + (self.highest_lambda - self.lowest_lambda) / self.periods[0] * offset
        elif offset <= self.periods[0] + self.periods[1]:
            return self.highest_lambda
        elif offset <= self.periods[0] + self.periods[1] + self.periods[2]:
            return self.highest_lambda + (self.lowest_lambda - self.highest_lambda) / self.periods[2] * (offset - self.periods[0] - self.periods[1])
        else:
            return self.lowest_lambda


if __name__ == '__main__':
    # Campaign A
    generator = DataGenerator(
        n_control=500,
        n_treated=500,
        mail_date=date(2022, 5, 15),
        before=14,
        after=28,
        lowest_lambda=0.5,
        highest_lambda=0.7,
        periods=(5, 2, 14),
    ).simple()
    generator.num_journeys.to_csv("campaign-a-num-journeys.csv", index=False)
    generator.spendings.to_csv("campaign-a-spendings.csv", index=False)

    # Campaign B
    generator = DataGenerator(
        n_control=1000,
        n_treated=2500,
        mail_date=date(2022, 6, 1),
        before=14,
        after=28,
        lowest_lambda=0.5,
        highest_lambda=0.7,
        periods=(7, 7, 10),
    ).simple()
    generator.num_journeys.to_csv("campaign-b-num-journeys.csv", index=False)
    generator.spendings.to_csv("campaign-b-spendings.csv", index=False)

    # Dummy
    generator = DataGenerator(
        n_control=500,
        n_treated=500,
        mail_date=date(2022, 1, 1),
        before=14,
        after=28,
        lowest_lambda=0.5,
        highest_lambda=0.55,
        periods=(5, 2, 14),
    ).simple()
    generator.num_journeys.to_csv("campaign-dummy-num-journeys.csv", index=False)
    generator.spendings.to_csv("campaign-dummy-spendings.csv", index=False)


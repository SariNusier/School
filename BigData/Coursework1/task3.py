from mrjob.job import MRJob, MRStep


class MRTopThreeAges(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   combiner=self.combiner,
                   reducer=self.reducer),
            MRStep(reducer=self.reducer_find_first_three)
        ]

    def mapper(self, _, line):
        if line:
            line = line.strip()
            age = line.split(",")[0]
            yield(age, 1)

    def combiner(self, key, values):
        yield(key, sum(values))

    def reducer(self, key, values):
        yield None, (sum(values), key)

    def reducer_find_first_three(self, _, age_count_pairs):
        age_count_pairs = sorted(list(age_count_pairs), reverse=True)
        age_count_pairs = age_count_pairs[:3]
        for age in age_count_pairs:
            yield age[1], age[0]


if __name__ == '__main__':
    MRTopThreeAges.run()
import os
from mrjob.job import MRJob, MRStep


class MRJoin(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer)
        ]

    def mapper(self, _, line):
        line = line.strip()

        # We select the id and the other 2 columns from the csv file
        uid, col1, col2 = [i.strip() for i in line.split(",")]

        # We get the filename
        file_name = os.environ['mapreduce_map_input_file']

        yield(uid, (file_name, (col1, col2)))

    def reducer(self, key, values):
        values = list(values)

        # Unpacking the tuple from the two files
        file1, file2 = values

        # Unpacking the values for each file
        file1_name, file1_values = file1
        file2_name, file2_values = file2

        # We return the values by writing age and occupation first
        # followed by education and marital status to keep
        # an even order regardless of the order of files inputted
        if file1_name == "id_age_occ.csv":
            yield(key, (file1_values, file2_values))
        else:
            yield(key, (file2_values, file1_values))


if __name__ == '__main__':
    MRJoin.run()

class RawData(object):
    def __init__(self,date,open,high,close,low,volume):
        self.date = date
        self.open = open
        self.high = high
        self.close = close
        self.low = low
        self.volume = volume


def read_sample_data(path):
    print("reading histories...")
    raw_data = []
    separator = "\t"
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("date"):  # ignore label line
                continue
            l = line[:-1]
            fields = l.split(separator)
            if len(fields) > 5:
                raw_data.append(RawData(fields[0], float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])))
    sorted_data = sorted(raw_data, key=lambda x: x.date)
    print("got %s records." % len(sorted_data))
    return sorted_data
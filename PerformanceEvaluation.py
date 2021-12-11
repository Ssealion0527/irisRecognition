def crr(L_result, test_label):
    """
        calculate crr

        Args:
            L_result: matching result
            test_label: true label for test set images

        Returns:
           crr
    """

    L_count = 0
    for i in range(len(L_result)):
        if L_result[i] == test_label[i]:
            L_count += 1
    return L_count / len(L_result)


def roc(L_result, L_distance, test_label, threshold):
    """
        calculate crr

        Args:
            L_result: matching result
            L_distance: matching distance
            test_label: true label for test set images
            threshold: a list of threshold

        Returns:
            fmr: a list of false match rate for each threshold
            fnmr: a list of false non-match rate for each threshold
    """
    positive = []
    negative = []
    for i in range(len(L_result)):
        if L_result[i] == test_label[i]:
            positive.append(L_distance[i])
        else:
            negative.append(L_distance[i])

    fmr = []
    fnmr = []
    for element in threshold:
        fm = 0
        fnm = 0
        # count the number of true matching with distance bigger than threshold
        for item in positive:
            if item > element:
                fnm += 1
        # count the number of false matching with distance smaller than threshold
        for item in negative:
            if item < element:
                fm += 1

        fmr.append(fm / len(negative) * 100)
        fnmr.append(fnm / len(positive) * 100)

    return fmr, fnmr
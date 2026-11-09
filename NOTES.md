Classification is a supervised machine learning task in which a model learns from labeled examples to assign new data points to one of a set of predefined categories, such as labeling a network flow as “attack” or “normal.”



I kept THRESHOLD = 0.4 because in an intrusion detection system a missed attack is worse than a false alarm: a false alarm only costs an analyst some time to check, while an undetected attack can compromise the network (with 0.4, recall rose from 0.50 to 0.86)


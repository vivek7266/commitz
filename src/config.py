class VectorizerConfig:
    NUM_FEATURES = 1000


class MetaConfig:
    BUGGY_KEYWORDS = ["bug", "fix", "wrong", "error", "fail", "problem", "patch"]


class LdaConfig:
    NUM_TOPICS = 2
    NUM_ITERATIONS = 20
    NUM_TOP_WORDS = 10
    NUM_TOP_WORDS_LARGE = 100


class Metrics:
    F_BETA_BETA_VALUE = 2


class SvmConfig:
    C_VALUE = 10

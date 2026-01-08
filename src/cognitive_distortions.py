"""
Cognitive Distortion Taxonomy and Definitions
Based on CBT (Cognitive Behavioral Therapy) principles
"""

# 12 Core Cognitive Distortions
COGNITIVE_DISTORTION_LABELS = [
    'all_or_nothing',          # 0
    'overgeneralization',      # 1
    'mental_filter',           # 2
    'discounting_positive',    # 3
    'mind_reading',            # 4
    'fortune_telling',         # 5
    'catastrophizing',         # 6
    'minimization',            # 7
    'emotional_reasoning',     # 8
    'should_statements',       # 9
    'labeling',                # 10
    'personalization',         # 11
    # Note: We exclude 'blame' as it's less common in self-reflection text
]

# Human-readable names
DISTORTION_NAMES = {
    'all_or_nothing': 'All-or-Nothing Thinking',
    'overgeneralization': 'Overgeneralization',
    'mental_filter': 'Mental Filter',
    'discounting_positive': 'Discounting the Positive',
    'mind_reading': 'Mind Reading',
    'fortune_telling': 'Fortune Telling',
    'catastrophizing': 'Catastrophizing',
    'minimization': 'Minimization',
    'emotional_reasoning': 'Emotional Reasoning',
    'should_statements': 'Should Statements',
    'labeling': 'Labeling',
    'personalization': 'Personalization',
}

# Short descriptions for each distortion
DISTORTION_DESCRIPTIONS = {
    'all_or_nothing': 'Seeing things in extreme categories with no middle ground',
    'overgeneralization': 'Making broad conclusions from single events',
    'mental_filter': 'Focusing exclusively on negatives, ignoring positives',
    'discounting_positive': 'Rejecting positive experiences as "not counting"',
    'mind_reading': 'Assuming you know what others think without evidence',
    'fortune_telling': 'Predicting negative outcomes with certainty',
    'catastrophizing': 'Expecting or imagining the worst-case scenario',
    'minimization': 'Inappropriately shrinking the importance of positive events',
    'emotional_reasoning': 'Believing feelings equal facts',
    'should_statements': 'Rigid rules about how you/others "should" behave',
    'labeling': 'Assigning global negative labels to self/others',
    'personalization': 'Blaming yourself for things outside your control',
}

# Signal words/phrases that indicate each distortion
SIGNAL_WORDS = {
    'all_or_nothing': [
        'always', 'never', 'completely', 'totally', 'perfect', 'everything',
        'everyone', 'every time', 'all or nothing', 'either or', 'no middle ground'
    ],
    'overgeneralization': [
        'always', 'never', 'everyone', 'no one', 'every time', 'all the time',
        'everybody', 'nobody', 'everything', 'nothing'
    ],
    'mental_filter': [
        'the only thing', 'just', 'only', 'but', 'however', 'even though',
        'despite', 'ignoring', 'forgetting', 'overlooking'
    ],
    'discounting_positive': [
        'but', 'however', 'even though', 'yes but', 'doesn\'t count',
        'not really', 'just luck', 'just being nice', 'doesn\'t matter'
    ],
    'mind_reading': [
        'they think', 'they must think', 'everyone thinks', 'people think',
        'they believe', 'they feel', 'they see me as', 'they view me as',
        'they probably think', 'they must believe'
    ],
    'fortune_telling': [
        'will definitely', 'going to', 'will never', 'will always',
        'is definitely going to', 'will surely', 'will certainly',
        'i know that', 'i can see that', 'it\'s clear that'
    ],
    'catastrophizing': [
        'terrible', 'awful', 'disaster', 'ruined', 'nightmare', 'horrible',
        'worst', 'end of the world', 'life is over', 'everything is ruined',
        'completely destroyed', 'impossible to recover'
    ],
    'minimization': [
        'not a big deal', 'doesn\'t matter', 'wasn\'t important', 'anyone could',
        'it was nothing', 'not significant', 'no big deal', 'small thing'
    ],
    'emotional_reasoning': [
        'i feel', 'i feel like', 'because i feel', 'since i feel',
        'i feel therefore', 'feeling means', 'my feelings tell me',
        'i know it\'s true because i feel'
    ],
    'should_statements': [
        'should', 'must', 'ought to', 'have to', 'need to', 'shouldn\'t',
        'must not', 'should have', 'must have', 'ought to have',
        'have to be', 'should be', 'must be'
    ],
    'labeling': [
        'i\'m a', 'i am a', 'they\'re a', 'he\'s a', 'she\'s a', 'it\'s a',
        'i\'m such a', 'i\'m just a', 'you\'re a', 'we\'re all',
        'loser', 'idiot', 'failure', 'stupid', 'worthless'
    ],
    'personalization': [
        'my fault', 'because of me', 'if i had', 'if i didn\'t',
        'i caused', 'i made', 'i should have', 'i shouldn\'t have',
        'it\'s my responsibility', 'i am responsible for'
    ],
}

# Example texts for each distortion (for synthetic data generation)
EXAMPLE_TEXTS = {
    'all_or_nothing': [
        "If I'm not perfect, I'm a total failure",
        "This project is completely ruined because of one mistake",
        "I either succeed at everything or I'm worthless",
        "Nothing ever goes right for me",
    ],
    'overgeneralization': [
        "I failed this test, I'm bad at everything",
        "No one ever likes me",
        "I always mess things up",
        "This always happens to me",
    ],
    'mental_filter': [
        "The whole day was terrible because of that one thing",
        "I got 9 compliments but only remember the one criticism",
        "Everything went wrong today",
        "Nothing good happened, only bad things",
    ],
    'discounting_positive': [
        "They only complimented me to be nice",
        "That success was just luck, not skill",
        "Anyone could have done that",
        "It doesn't count because it was easy",
    ],
    'mind_reading': [
        "She thinks I'm stupid",
        "Everyone at the party thought I was awkward",
        "They must think I'm incompetent",
        "People probably see me as a failure",
    ],
    'fortune_telling': [
        "I'm definitely going to fail this interview",
        "This relationship will never work out",
        "I know I'm going to embarrass myself",
        "It's clear that this will end badly",
    ],
    'catastrophizing': [
        "If I fail this exam, my entire life is ruined",
        "This headache is probably a brain tumor",
        "Everything is completely destroyed now",
        "This is the worst possible thing that could happen",
    ],
    'minimization': [
        "Getting that promotion wasn't a big deal",
        "Anyone could have done that",
        "It was nothing special",
        "My achievements don't really matter",
    ],
    'emotional_reasoning': [
        "I feel stupid, therefore I am stupid",
        "I feel anxious, so something bad must be about to happen",
        "Because I feel worthless, I must be worthless",
        "My feelings tell me this is a disaster",
    ],
    'should_statements': [
        "I should be able to handle this",
        "They should have known better",
        "I must be perfect at everything",
        "I shouldn't make mistakes",
    ],
    'labeling': [
        "I'm a loser because I made a mistake",
        "He's an idiot for doing that",
        "I'm just a failure",
        "I'm worthless",
    ],
    'personalization': [
        "My team lost because I wasn't supportive enough",
        "It's my fault my parents argue",
        "I caused this whole problem",
        "Everything went wrong because of me",
    ],
}

# Neutral examples (no distortions)
NEUTRAL_EXAMPLES = [
    "I had a good day today. Work went well and I got some exercise.",
    "The weather was nice this morning. I went for a walk in the park.",
    "I'm learning a new skill. It takes time but I'm making progress.",
    "I had a conversation with my friend. We discussed our plans for the weekend.",
    "I completed my tasks today. Some went well, others need improvement.",
    "I'm feeling tired after a long day. I'll rest and recharge.",
    "I tried a new recipe today. It turned out okay, I'll adjust it next time.",
    "I'm planning my vacation. There are many options to consider.",
]


def get_distortion_info(distortion_id):
    """
    Get information about a cognitive distortion
    
    Args:
        distortion_id: Integer index or string label
        
    Returns:
        dict with name, description, signal_words, examples
    """
    if isinstance(distortion_id, int):
        if 0 <= distortion_id < len(COGNITIVE_DISTORTION_LABELS):
            label = COGNITIVE_DISTORTION_LABELS[distortion_id]
        else:
            return None
    elif isinstance(distortion_id, str):
        label = distortion_id
    else:
        return None
    
    if label not in COGNITIVE_DISTORTION_LABELS:
        return None
    
    return {
        'label': label,
        'name': DISTORTION_NAMES[label],
        'description': DISTORTION_DESCRIPTIONS[label],
        'signal_words': SIGNAL_WORDS[label],
        'examples': EXAMPLE_TEXTS[label],
    }


def get_all_distortions():
    """Get list of all distortion labels"""
    return COGNITIVE_DISTORTION_LABELS.copy()


def get_num_distortions():
    """Get number of distortion types"""
    return len(COGNITIVE_DISTORTION_LABELS)

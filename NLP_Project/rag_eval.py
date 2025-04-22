from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, HallucinationMetric


ip = '''\
My friend invited me to a party this weekend. I said yes but now my chest feels tight.
I keep imagining walking in and everyone staring. My hands won't stop sweating.
Checked the event page 20 times today. What if I don't know anyone there?
Woke up nauseous and couldn’t eat breakfast. My heart races when I think about leaving home.
Practiced introductions in the mirror for an hour. Voice kept shaking during the rehearsal.
Texted my friend three different excuses to cancel but deleted them all. Hands trembling.
Arrived 30 minutes late and sat in my car crying. Couldn’t make myself open the door.
Drove home without going in. Now I’m beating myself up for being weak.
Can’t stop thinking people will judge me for not showing up. Checked social media all morning.
Headache won’t go away, and I keep pacing around my apartment. Feel like I failed some test.
The mirror practice didn’t help. How do I stop the physical symptoms?'''
ip = ip.split('\n')

op = '''\
I'm sorry the mirror practice didn't help as much as you had hoped. Stopping physical symptoms like sweating, trembling, or rapid heartbeat can be challenging, but there are some techniques you can try: Deep breathing exercises, progressive muscle relaxation, and visualization might help calm your body. Additionally, consider practicing mindfulness meditation to become more aware of your physical sensations. When you start feeling anxious, try acknowledging the physical sensations without judgment, and gently redirect your focus on a calming thought or sensation. It's also essential to take care of yourself by getting enough sleep, exercising regularly, and eating a balanced diet. If these strategies don't help significantly, it might be helpful to consult with a mental health professional for personalized guidance.
I remember you trying the mirror practice to manage your physical symptoms of anxiety, and unfortunately, it didn't have the desired effect. If you're open to exploring other options, there are a few techniques that might be helpful. For instance, some people find deep breathing exercises or progressive muscle relaxation to be effective in reducing physical symptoms like rapid heartbeat or sweating. Would you like me to share more about these approaches?'''
op = op.split('\n')


def measure(input_str, actual_output_str, context_list):
    # Create a test case
    test_case = LLMTestCase(
        input=input_str,
        actual_output=actual_output_str,
        context=context_list,
        retrieval_context=context_list
    )

    # Initialize the metrics
    answer_relevancy = AnswerRelevancyMetric()
    context_relevancy = ContextualRelevancyMetric()
    faithfulness = FaithfulnessMetric()
    hallucination = HallucinationMetric()

    # Measure each metric
    answer_relevancy.measure(test_case)
    context_relevancy.measure(test_case)
    faithfulness.measure(test_case)
    hallucination.measure(test_case)

    # Print the results
    print(f"Answer Relevancy: Score={answer_relevancy.score}, Reason={answer_relevancy.reason}")
    print(f"Context Relevancy: Score={context_relevancy.score}, Reason={context_relevancy.reason}")
    print(f"Faithfulness: Score={faithfulness.score}, Reason={faithfulness.reason}")
    print(f"Hallucination: Score={hallucination.score}, Reason={hallucination.reason}")
    print()

    with open('output.txt', 'a') as f:
        f.write(f'{answer_relevancy.score}\n{context_relevancy.score}\n{faithfulness.score}\n{hallucination.score}\n\n')


context_list = ip[:10]

input_str = ip[10]
actual_output_str = op[0]
measure(input_str, actual_output_str, context_list)

input_str = ip[10]
actual_output_str = op[1]
measure(input_str, actual_output_str, context_list)


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Define input and output variables
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# 2. Generate membership functions
quality.automf(3)
service.automf(3)

# 3. Define membership functions for tip
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# 4. Visualize membership functions
quality.view()
service.view()
tip.view()

# 5. Define fuzzy rules
rule1 = ctrl.Rule(
    quality['poor'] | service['poor'],
    tip['low']
)

rule2 = ctrl.Rule(
    service['average'],
    tip['medium']
)

rule3 = ctrl.Rule(
    quality['good'] | service['good'],
    tip['high']
)

# 6. Visualize rules
rule1.view()

# 7. Create fuzzy control system
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# 8. Give input values
tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

# 9. Calculate output
tipping.compute()

# 10. Display result
print("Recommended Tip:", tipping.output['tip'])

# 11. Visualize output
tip.view(sim=tipping)

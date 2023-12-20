import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def flame_set_up(self, scale, color=(1, 0.1, 0.2)):
    flame_base = self.rocket_width / 2
    self.half_flame_base = flame_base / 2
    self.flame_height = self.tank_height * 1.2

    vertices = [
        [
            self.center_tank[0] - self.half_flame_base + self.rocket_width / 2,
            self.center_tank[1] - self.combustion_chamber_height * 1.4,
        ],
        [
            self.center_tank[0] + self.half_flame_base + self.rocket_width / 2,
            self.center_tank[1] - self.combustion_chamber_height * 1.4,
        ],
        [
            self.center_tank[0] + self.rocket_width / 2,
            -(self.flame_height / 2) * scale - 1,
        ],
    ]

    return patches.Polygon(vertices, closed=True, color=color)


##Flame fits##
mean_fit = [
    3.89727126e-01,
    -9.41821451e00,
    8.25958346e01,
    -3.12920916e02,
    4.44644581e02,
    -7.87109541e01,
]
std_fit = [
    5.79196252e-02,
    -1.30279120e00,
    9.99919598e00,
    -2.56161052e01,
    -2.74165194e01,
    3.31486104e02,
]


def update_flame(self, thrust, scale, flame):
    vertices = flame.xy

    mean_error_point = np.polyval(mean_fit, self.t)
    std_error_point = np.polyval(std_fit, self.t)

    oscillations = np.random.normal(loc=mean_error_point, scale=std_error_point)
    flame_percent = (
        thrust
        + (1 if np.random.random() < 0.5 else -1)
        * oscillations
        * self.scale_oscillations
    ) / self.max_thrust

    vertices[2] = [
        vertices[2][0],
        -1 - scale * self.flame_height * flame_percent,
    ]

    flame.set_xy(vertices)


def update_grain(self, r):
    fuel_percent = 1 - r / self.r_final
    self.fuel1.set_width(self.fuel_width * fuel_percent)
    self.fuel2.set_width(self.fuel_width * fuel_percent)
    self.fuel2.set_x(self.fuel_pos2[0] + (1 - fuel_percent) * self.fuel_width)


def update_ox(self, ox_mass):
    ox_percent = ox_mass / self.max_ox
    self.ox.set_height(self.tank_height * ox_percent - 0.08 * (1 - ox_percent))


class motor_animation:
    def __init__(self):
        self.t = 0
        self.rocket_width = 0.3
        self.center_tank = 0.5, 0.5
        self.tank_height = 1.2
        self.combustion_chamber_height = 0.85
        self.wall_width = 0.01
        self.fuel_width = self.rocket_width * 0.7
        self.fuel_shift = self.rocket_width - self.fuel_width - self.wall_width
        self.max_thrust = 4725  # determined with the previous sim
        self.distance_tank_comb = 0.1
        self.final_radius = 0.2

        self.scale_oscillations = 3

    def set_up(self, ox_max, r_max, ri):
        anim, motor_plot = plt.subplots(1, 1, figsize=(2, 7))

        combustion_chamber_height = (
            self.center_tank[0]
            - self.combustion_chamber_height
            - self.distance_tank_comb
        )

        tank = patches.FancyBboxPatch(
            self.center_tank,
            self.rocket_width,
            self.tank_height,
            boxstyle="round,pad=0.05",
            facecolor="0.7",
        )

        combustion_chamber = patches.Rectangle(
            (self.center_tank[0], combustion_chamber_height),
            self.rocket_width,
            self.combustion_chamber_height,
            facecolor="0.2",
        )

        ox = patches.FancyBboxPatch(
            (self.center_tank[0] + self.wall_width, self.center_tank[0]),
            self.rocket_width - 2 * self.wall_width,
            self.tank_height - 0.03,
            boxstyle="round,pad=0.04",
            facecolor="blue",
        )

        fuel_pos1 = (
            self.center_tank[0] + self.wall_width,
            combustion_chamber_height + self.wall_width,
        )

        self.fuel_pos2 = (
            self.center_tank[0] + self.fuel_shift,
            combustion_chamber_height + self.wall_width,
        )
        fuel_color = (1, 0.35, 0)
        fuel1 = patches.Rectangle(
            fuel_pos1,
            self.fuel_width,
            self.combustion_chamber_height * 0.9,
            facecolor=fuel_color,
        )
        fuel2 = patches.Rectangle(
            self.fuel_pos2,
            self.fuel_width,
            self.combustion_chamber_height * 0.9,
            facecolor=fuel_color,
        )

        circle_center = 1.2, -0.1
        final_radius = 0.2
        self.combustion_top = patches.Circle(
            circle_center,
            radius=final_radius + 0.01,
            linewidth=1,
            edgecolor="0.2",
            facecolor=(1, 0.35, 0),
        )
        self.grain = patches.Circle(circle_center, radius=0, color="0.2")

        self.flame = flame_set_up(self, scale=1, color=(1, 0.5, 0))
        self.interior_flame = flame_set_up(self, scale=0.5, color=(1, 0.1, 0.2))
        self.ox = ox
        self.max_ox = ox_max
        self.r_final = r_max
        self.ri = ri
        self.fuel1 = fuel1
        self.fuel2 = fuel2

        patchess = [
            tank,
            combustion_chamber,
            self.ox,
            self.fuel1,
            self.fuel2,
            self.flame,
            self.interior_flame,
            self.combustion_top,
            self.grain,
        ]
        for patche in patchess:
            motor_plot.add_patch(patche)

        plt.ion()
        plt.show()
        plt.ylim(-4, 1.8)
        plt.xlim(0.4, 1.5)
        motor_plot.set_aspect("equal")
        motor_plot.axis("off")
        self.anim = anim, motor_plot

    def step(self, ox_mass, r, thrust, dt=0.001, fps=60):
        self.t += dt
        # round 1/dt in case odd choice

        size = round(1 / dt) // fps
        if (self.t // dt) % size == 0:
            self.anim[0].canvas.flush_events()
            self.anim[0].canvas.draw_idle()
            #self.anim[0].savefig(f"ANIMATION/plot_{self.t}.png")

            update_grain(self, r)
            update_ox(self, ox_mass)

            if thrust != 0:
                update_flame(self, thrust, scale=1, flame=self.flame)
                update_flame(self, thrust, scale=0.4, flame=self.interior_flame)

            self.grain.radius = self.final_radius * r / self.r_final

        return (
            self.ox,
            self.fuel1,
            self.fuel2,
            self.flame,
            self.interior_flame,
            self.grain,
        )

    def save(self, filename):
        # Make sure the animation attribute exists and is not None
        if hasattr(self, "anim") and self.anim:
            # Save the animation
            self.anim[0].savefig(filename)
            plt.show(block=True)
        else:
            print("Animation has not been created yet or is None.")

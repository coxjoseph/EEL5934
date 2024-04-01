import numpy as np
import matplotlib.pyplot as plt


def elementwise_array_add(x: list, y: list) -> list:
    """
    Print the elements of y added to each element of x. Answer to question 1C, part b

    > x = elementwise_array_add([1, 2, 3], [1])
    [2, 3, 4]
    > x
    [[2, 3, 4]]
    :param x: The first array, a list of numbers
    :param y: The second array, a list of numbers
    :return: A list of lists, where the i-th element in the list is the i-th element in y added to elements of x
    """
    # List comprehension (fancy for loop) for addition
    o = [[j + i for j in x] for i in y]

    # Print to console
    for array in o:
        print(array)

    return o


def make_circle(cent: tuple, rad: int) -> np.ndarray:
    """
    Generate circle function. Python implementation of the MakeCircle.m function found in the homework. Creates a
    circle centered at center with a radius of radius in a 101x101 array.
    :param cent: center of the circle as [x, y]
    :param rad: radius of the circle
    :return: a logical mask (type int) with value 1 if the defined circle is in the boundaries and 0 if not
    """
    # Create a 101x101 array with zeros
    image = np.zeros((101, 101), dtype=int)

    # Generate coordinates for the circle
    y, x = np.ogrid[:101, :101]
    dist_from_center = np.sqrt((x - cent[0]) ** 2 + (y - cent[1]) ** 2)

    # Set values inside the circle to 1
    image[dist_from_center <= rad] = 1

    return image


if __name__ == '__main__':
    # Question 1C, part b:
    elementwise_array_add([2, 4, 5, 6], [1, 3, 5])

    # Question 2B:
    center = (25, 25)
    radius = 15
    circle_mask = make_circle(center, radius)

    plt.imshow(circle_mask, cmap='gray', origin='lower')
    plt.title('Generated Circle')
    plt.show()

    # Question 2C:
    # Create heatmap
    num_circles = 500
    heatmap = np.zeros((101, 101), dtype=int)
    for _ in range(num_circles):
        center = np.random.randint(0, 101, size=2)  # Random center coordinates
        radius = np.random.randint(5, 25)           # Random radius between 5 and 5
        heatmap += make_circle(center, radius)

    # Display the heatmap
    plt.imshow(heatmap, cmap='hot', origin='lower')
    plt.title('Heatmap of 500 Circles')
    plt.colorbar()
    plt.show()

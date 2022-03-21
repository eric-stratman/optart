from collections import defaultdict
import cv2
import imageio as iio
import numpy as np
import gurobipy as gp
from PIL import Image
import os


class DominoSolver:
    def __init__(self, s, columns, rows, image_path, pixels_per_domino=1):
        """
        A complete set of double-nine dominoes contains 55 dominoes,
        partition the target image and the canvas into m rows and n columns such that mn=110s.
        @param s:Num of domionos set
        @param width: Number of rows
        @param height: Number of columns
        """
        self.domino_image = None
        self.s = s
        self.rows = rows
        self.columns = columns
        if columns * rows / s != 110:
            raise Exception("Make sure that (m*n)/s=110")
        self.pixels_per_domino = pixels_per_domino
        self.image_path = image_path
        self.original_image = iio.imread(image_path)
        self.rescaled_image = None
        # pairs of adjacent squares
        self.P = None  # placeholder
        self.P_query = defaultdict(
            set)  # placeholder for querying pairs as per row and column--needed in model building
        self.D = None  # placeholder
        self.cost = gp.tupledict()
        # gurobi placeholders
        self.model = gp.Model()
        self.x = {}
        # output placeholders
        self.matchings = []  # (d1,d2) and (d2,d1) are treated as the same by the model
        self.orientations = []  # distinction between (d1,d2) and (d2,d1)

    def fit(self):
        self.preprocess_image()
        self.build_pairs()
        self.calculate_cost()
        self.build_model()
        self.solve_model()
        self.set_orientation()
        self.build_domino_image()

    def preprocess_image(self):
        """
        scale down image to m by n pixels
        @return:
        """
        print("-Converting image to B&W and rescaling")
        grayImage = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # restrict pixels to -0.5 to 9.5
        grayImage = np.interp(grayImage, (grayImage.min(), grayImage.max()), (-0.5, 9.5))
        grayImage = cv2.resize(grayImage,
                               dsize=(1 * self.columns, 1 * self.rows),
                               interpolation=cv2.INTER_CUBIC)
        self.rescaled_image = grayImage
        print("  --Done")

    def build_pairs(self):
        # set P
        p = set()
        for i in range(0, self.columns - 1):
            for j in range(0, self.rows):
                pair = ((i, j), (i + 1, j))  # horizontal
                p.add(pair)
                self.P_query[i, j].add(pair)
                self.P_query[i + 1, j].add(pair)
        for i in range(0, self.columns):
            for j in range(0, self.rows - 1):
                pair = ((i, j), (i, j + 1))  # vertical
                p.add(pair)
                self.P_query[i, j].add(pair)
                self.P_query[i, j + 1].add(pair)
        self.P = gp.tuplelist(list(p))
        # set D
        d = set()
        for d1 in range(10):
            for d2 in range(10):
                if d2 >= d1:
                    d.add((d1, d2))
        self.D = gp.tuplelist(list(d))

    def calculate_cost(self):
        """
        calculate cost of assigning domino to a particular position based on brightness
        @return:
        """
        print("-Calculating cost")
        for d in self.D:
            d1, d2 = d[0], d[1]
            for p in self.P:
                col1, row1, col2, row2 = p[0][0], p[0][1], p[1][0], p[1][1]
                c = min((d1 - self.rescaled_image[row1, col1]) ** 2 + (d2 - self.rescaled_image[row2, col2]) ** 2,
                        (d1 - self.rescaled_image[row2, col2]) ** 2 + (d2 - self.rescaled_image[row1, col1]) ** 2
                        )
                self.cost[d, p] = c
        print("  --Done")

    def build_model(self):
        """

        @return:
        """
        print("-Building model")
        for d in self.D:
            for p in self.P:
                self.x[d, p] = self.model.addVar(vtype=gp.GRB.BINARY)
        # self.x = self.model.addVars(self.D, self.P, obj=self.cost)
        # ensure each domino is placed on the canvas
        self.model.addConstrs(gp.quicksum(self.x[d, p] for p in self.P) == self.s for d in self.D)
        # each square is covered by exactly one domino
        for i in range(0, self.columns):
            for j in range(0, self.rows):
                # get subset of P containing i,j
                p_subset = list(self.P_query[i, j])
                self.model.addConstr(gp.quicksum(self.x[d, p] for p in p_subset for d in self.D) == 1)
        # self.model.ModelSense = 1  # minimize
        self.model.setObjective(gp.quicksum(self.x[d, p] * self.cost[d, p] for d in self.D for p in self.P),
                                gp.GRB.MINIMIZE)
        self.model.update()
        print("  --Done")

    def solve_model(self, verbose=0):
        print("-Solving model")
        self.model.setParam('OutputFlag', verbose)
        self.model.optimize()
        sol_x = self.model.getAttr('x', self.x)
        for k, v in sol_x.items():
            if v > 0.99:
                self.matchings.append(k)
        print("  --Done")

    def set_orientation(self):
        """
        decide which side does the domino go (d1,d2) or (d2,d1)
        @return:
        """
        if len(self.matchings) == 0:
            raise Exception("Make sure to solve the model before setting the orientation")
        for m in self.matchings:
            d = m[0]
            d1, d2 = d[0], d[1]
            p = m[1]
            col1, row1, col2, row2 = p[0][0], p[0][1], p[1][0], p[1][1]
            temp1 = (d1 - self.rescaled_image[row1, col1]) ** 2 + (d2 - self.rescaled_image[row2, col2]) ** 2
            temp2 = (d1 - self.rescaled_image[row2, col2]) ** 2 + (d2 - self.rescaled_image[row1, col1]) ** 2
            if temp1 <= temp2:
                self.orientations.append(((d1, d2), p))
            else:
                self.orientations.append(((d2, d1), p))

    def build_domino_image(self):
        """
        Put dominoes together based on orientation
        @return:
        """
        print("\n-Building domino image")
        pixels_per_domino = self.pixels_per_domino
        # create a background and paste domino over it using PIL library
        domino_image = Image.new('RGBA', (self.pixels_per_domino * self.columns, self.pixels_per_domino * self.rows))
        for o in self.orientations:
            d, p = o[0], o[1]
            # d tells which domino to use and p tells orientation
            # x1, x2 = p[0][0], p[1][0]
            # y1, y2 = p[0][1], p[1][1]
            col1, row1, col2, row2 = p[0][0], p[0][1], p[1][0], p[1][1]
            # check if domino d at position p is horizontal or vertical
            if row1 == row2:
                horizontal = True
            else:
                horizontal = False
            domino_file = str(d[0]) + "-" + str(d[1]) + ".png"
            # all domino images are vertical in the directory
            single_domino_img = Image.open(f'./dominoes/{domino_file}', 'r')
            single_domino_img = single_domino_img.resize(
                (1 * pixels_per_domino, 2 * pixels_per_domino), Image.NEAREST)  # dominoes are of ratio 1:2
            if horizontal:
                # rotates counter clockwise
                single_domino_img = single_domino_img.rotate(90, expand=True)
            # give upper left corner 
            position = (col1 * pixels_per_domino, row1 * pixels_per_domino)
            domino_image.paste(single_domino_img, position)
        output_path = './domino_output/'
        if not os.path.isdir('./domino_output'):
            os.makedirs(output_path)
        filename = self.image_path.split("/")[-1].split(".")[0] + '-domino' + '.png'
        domino_image.save(output_path + filename)
        self.domino_image = domino_image
        print("Task completed, Image saved as {}".format(output_path + filename))


if __name__ == '__main__':
    print("--Run model")
    # make sure rows x columns = 110 x s
    domi = DominoSolver(s=12, rows=44, columns=30, image_path='./input_images/ts.jpeg',
                        pixels_per_domino=200)
    domi.fit()

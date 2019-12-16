%% MyMainScript

tic;
%% Your code here
clc
clear all;

D = 40;
img = imread('../data/barbara256.png');
myIdealFilt(D, img);
D = 80;
img = imread('../data/barbara256.png');
myIdealFilt(D, img);

sigma = 40;
img = imread('../data/barbara256.png');
myGaussianFilt(sigma, img);
sigma = 80;
img = imread('../data/barbara256.png');
myGaussianFilt(sigma, img);

toc;

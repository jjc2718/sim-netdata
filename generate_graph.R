library(argparse)
library(MASS)
source('community_graph.R')

generate_data <- function(args) {

    # convert adjacency matrix to Graph object
    A <- read.table(args$adjacency_matrix, sep='\t')
    d <- dim(A)[1]
    print(d)

    # G <- adj2graph(A)
    # print(G)

    # fit precision matrix to graph (TODO: explain how this works)
    # see https://stats.stackexchange.com/q/295093 for basics
    # P <- graph2prec(G)

    # get covariance matrix (inverse of precision matrix)
    # Sigma <- solve(P)

    # generate data from MVN with 0 mean and given covariance
    # X <- mvrnorm(n=n, rep(0, d), Sigma)

    # write data to file
    # write.table(X, file=args$output_file, sep='\t', row.names=F, col.names=F)

}

main <- function() {
    parser <- ArgumentParser(description='Script to generate data with network covariance')
    parser$add_argument('--adjacency_matrix', required=T,
                        help='File containing adjacency matrix of network to use')
    parser$add_argument('--n', default=100,
                        help='Number of data points to generate')
    parser$add_argument('--output_file', required=T,
                        help='Output file for generated data')
    args <- parser$parse_args()
    generate_data(args)
}

main()


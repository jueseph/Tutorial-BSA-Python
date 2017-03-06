#!/bin/bash
#
# Takes an alignment of reads to a reference genome and calls variants (SNPs and indels). 
# Part of Springer Lab pipeline for bulk segregant analysis in yeast.
#
# Usage:
#
#	./callvar_4sample.sh output_file parent1.bam parent2.bam segregant_pool1.bam segregant_pool2.bam
#
#
# Can also be given fewer or more than 4 samples as needed.
# 
# Before running this, run map_reads.sh to align raw read data to genome.
#
# 20170124 Updated, Jue Wang
#

ref="ref/sacCer3"

samtools mpileup -t DPR -ugf $ref.fa ${@:2} | bcftools call -vmO v -o $1.vcf

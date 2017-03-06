#!/bin/bash
#
# Takes sequencing reads from FASTQ and aligns to yeast reference genome, then compresses 
# alignment to .bam format, sorts, and indexes. Does not do read trimming because bwa mem 
# doesn't need it.
#
# Part of Springer Lab pipeline for bulk segregant analysis in yeast.
#
# Usage:
#
#	./map_reads.sh raw_reads.fastq
#
#	OR (for paired-end)
#
#	./map_reads.sh raw_reads_R1.fastq raw_reads_R2.fastq
#
# 
# Before running this, index the reference genome:
#
#	bwa index sacCer3.fa
# 
# After running this, run callvar_4sample.sh to identify SNPs and indels.
#
# 20170124 Updated, Jue Wang
#

ref="ref/sacCer3"
tempdir="temp"

filename=${1##*/}
filestub=${filename%%.*}

mkdir $tempdir

bwa mem $ref.fa $@ > $filestub.sam

samtools view -bS -o $filestub.unsorted.bam $filestub.sam

samtools sort -T $tempdir/$filestub -o $filestub.bam $filestub.unsorted.bam

samtools index $filestub.bam
